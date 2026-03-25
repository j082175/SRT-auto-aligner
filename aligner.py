"""
WhisperX 기반 자막 생성 및 재정렬 모듈.

모드 1 - 생성+정렬: 영상 → Whisper 전사 → wav2vec2 alignment → SRT
모드 2 - 정렬만:   영상 + 기존 SRT → wav2vec2 alignment → SRT

출력 파일명: {기본경로}.{언어코드}.srt  (예: movie.ko.srt)
"""

import os
import re
import tempfile
from typing import Any, Callable, List, Optional

import ffmpeg
import torch
import whisperx

from srt_utils import SRTSegment, parse_srt, write_srt

LANGUAGE_OPTIONS = {
    "자동 감지": None,
    "한국어": "ko",
    "영어": "en",
    "일본어": "ja",
    "중국어 (간체)": "zh",
    "스페인어": "es",
    "프랑스어": "fr",
    "독일어": "de",
    "러시아어": "ru",
    "포르투갈어": "pt",
    "이탈리아어": "it",
    "아랍어": "ar",
}

MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# 언어 코드 → spaCy 모델명
_SPACY_MODELS = {
    "en": "en_core_web_sm",
    "ko": "ko_core_news_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ru": "ru_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
}

# 줄 끝에 혼자 남으면 어색한 단어들
_DANGLING_WORDS = {
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "by", "up", "as", "with",
    "or", "and", "but", "nor", "so", "yet",
    "if", "not", "no",
}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "float32"


def build_output_path(base_srt_path: str, lang_code: str) -> str:
    root, ext = os.path.splitext(base_srt_path)
    return f"{root}.{lang_code}{ext}"


def extract_audio(input_path: str, output_wav: str) -> None:
    (
        ffmpeg
        .input(input_path)
        .output(output_wav, ar=16000, ac=1, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


def load_spacy_model(lang_code: str, log: Callable[[str], None] = print) -> Optional[Any]:
    """spaCy 모델 로드. 미설치 시 None 반환."""
    model_name = _SPACY_MODELS.get(lang_code)
    if not model_name:
        return None
    try:
        import spacy
        return spacy.load(model_name, disable=["ner", "lemmatizer"])
    except OSError:
        log(f"spaCy '{model_name}' 미설치 → 기본 분할 방식 사용.")
        log(f"  더 정확한 분할: python -m spacy download {model_name}")
        return None
    except ImportError:
        return None


def _get_noun_chunk_spans(text: str, nlp: Any) -> List[tuple]:
    """명사구 (start_char, end_char) 목록."""
    try:
        doc = nlp(text)
        return [(chunk.start_char, chunk.end_char) for chunk in doc.noun_chunks]
    except Exception:
        return []


def _is_safe_cut(cut_char: int, noun_spans: List[tuple]) -> bool:
    """cut_char 위치가 명사구 중간이 아닌지 확인."""
    for span_start, span_end in noun_spans:
        if span_start < cut_char < span_end:
            return False
    return True


def _split_words_smart(
    words: List[dict],
    seg_end: float,
    max_chars: int,
    noun_spans: List[tuple],
) -> List[dict]:
    """
    word 타임스탬프 기준 분할.
    명사구 중간에서 자르지 않도록 cut point를 조정.
    """
    word_texts = [w.get("word", "").strip() for w in words]

    # 각 단어의 문자 오프셋 계산 (joined text 기준)
    char_starts: List[int] = []
    char_ends: List[int] = []
    pos = 0
    for wt in word_texts:
        char_starts.append(pos)
        char_ends.append(pos + len(wt))
        pos += len(wt) + 1

    chunks: List[dict] = []
    start_idx = 0

    while start_idx < len(words):
        # 남은 텍스트가 max_chars 이하면 그냥 묶기
        remaining = " ".join(word_texts[start_idx:])
        if len(remaining) <= max_chars:
            chunk_words = words[start_idx:]
            chunks.append({
                "start": chunk_words[0]["start"],
                "end": seg_end,
                "text": remaining,
                "words": chunk_words,
            })
            break

        # max_chars를 초과하는 첫 번째 단어 인덱스 찾기
        overflow_idx = start_idx
        for i in range(start_idx, len(words)):
            chunk_len = char_ends[i] - char_starts[start_idx]
            if chunk_len > max_chars:
                overflow_idx = i
                break
        else:
            overflow_idx = len(words)

        # overflow_idx 직전부터 역방향으로 안전한 cut point 탐색
        cut_idx = None
        for i in range(overflow_idx - 1, start_idx, -1):
            cut_char = char_ends[i]
            if _is_safe_cut(cut_char, noun_spans):
                cut_idx = i
                break

        # 역방향에서 못 찾으면 overflow_idx 이후 정방향 탐색
        if cut_idx is None:
            for i in range(overflow_idx, len(words)):
                cut_char = char_ends[i]
                if _is_safe_cut(cut_char, noun_spans):
                    cut_idx = i
                    break

        # 그래도 없으면 그냥 overflow 직전에서 자름
        if cut_idx is None:
            cut_idx = max(overflow_idx - 1, start_idx)

        chunk_words = words[start_idx: cut_idx + 1]
        chunks.append({
            "start": chunk_words[0]["start"],
            "end": chunk_words[-1].get("end", chunk_words[-1]["start"] + 0.3),
            "text": " ".join(word_texts[start_idx: cut_idx + 1]),
            "words": chunk_words,
        })
        start_idx = cut_idx + 1

    return chunks


def _fix_dangling(chunks: List[dict]) -> List[dict]:
    """청크 끝 단어가 관사/전치사/접속사면 다음 청크 앞으로 이동."""
    if len(chunks) < 2:
        return chunks

    result = [dict(c) for c in chunks]

    for i in range(len(result) - 1):
        cur_words = result[i].get("words", [])
        if not cur_words:
            continue

        last_word = cur_words[-1].get("word", "").strip().rstrip(".,!?")
        if last_word.lower() not in _DANGLING_WORDS:
            continue

        move_word = cur_words[-1]
        next_words = [move_word] + result[i + 1].get("words", [])

        result[i]["words"] = cur_words[:-1]
        result[i]["text"] = " ".join(w.get("word", "").strip() for w in result[i]["words"])
        result[i]["end"] = (
            cur_words[-2].get("end", result[i]["end"]) if len(cur_words) > 1 else result[i]["start"]
        )
        result[i + 1]["words"] = next_words
        result[i + 1]["text"] = " ".join(w.get("word", "").strip() for w in next_words)
        result[i + 1]["start"] = move_word.get("start", result[i + 1]["start"])

    return [c for c in result if c.get("text", "").strip()]


def split_long_segments(
    aligned_segments: List[dict],
    max_chars: int = 42,
    nlp: Optional[Any] = None,
) -> List[dict]:
    """
    긴 세그먼트 분할.
    - nlp 제공 시: 명사구 경계 인식으로 스마트 분할
    - nlp 없을 시: 글자 수 기준 + dangling 보정
    word 타임스탬프 없으면 문장 부호 기준 + 글자 수 비례 시간 배분.
    """
    result = []
    for seg in aligned_segments:
        text = seg.get("text", "").strip()
        seg_start = seg.get("start")
        seg_end = seg.get("end")

        if seg_start is None or seg_end is None or len(text) <= max_chars:
            result.append(seg)
            continue

        words = [w for w in seg.get("words", []) if w.get("start") is not None]

        if words:
            # 명사구 spans 계산 (joined word text 기준)
            joined = " ".join(w.get("word", "").strip() for w in words)
            noun_spans = _get_noun_chunk_spans(joined, nlp) if nlp else []

            chunks = _split_words_smart(words, seg_end, max_chars, noun_spans)
            result.extend(_fix_dangling(chunks) if chunks else [seg])

        else:
            # word 타임스탬프 없음 → 문장 부호 기준 + 비례 분할
            parts = re.split(r'(?<=[.!?,])\s+', text)
            chunks_text: List[str] = []
            cur = ""
            for part in parts:
                candidate = (cur + " " + part).strip() if cur else part
                if cur and len(candidate) > max_chars:
                    chunks_text.append(cur)
                    cur = part
                else:
                    cur = candidate
            if cur:
                chunks_text.append(cur)

            if len(chunks_text) <= 1:
                result.append(seg)
                continue

            total_chars = sum(len(c) for c in chunks_text)
            duration = seg_end - seg_start
            cur_start = seg_start
            for i, chunk_text in enumerate(chunks_text):
                ratio = len(chunk_text) / total_chars
                chunk_end = (cur_start + duration * ratio) if i < len(chunks_text) - 1 else seg_end
                result.append({
                    "start": cur_start,
                    "end": chunk_end,
                    "text": chunk_text,
                    "words": [],
                })
                cur_start = chunk_end

    return result


def _wx_segments_to_srt(wx_segments: List[dict]) -> List[SRTSegment]:
    result = []
    for i, seg in enumerate(wx_segments, start=1):
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "").strip()
        if start is None or end is None or not text:
            continue
        result.append(SRTSegment(index=i, start=start, end=end, text=text))
    return result


def _merge_with_original_duration(
    aligned_segments: List[dict],
    original_segments: List[SRTSegment],
) -> List[SRTSegment]:
    new_starts: List[Optional[float]] = []
    aligned_idx = 0
    for _ in original_segments:
        if aligned_idx < len(aligned_segments):
            new_starts.append(aligned_segments[aligned_idx].get("start"))
            aligned_idx += 1
        else:
            new_starts.append(None)

    result = []
    for i, orig_seg in enumerate(original_segments):
        new_start = new_starts[i]
        duration = orig_seg.end - orig_seg.start

        if new_start is None:
            result.append(orig_seg)
            continue

        new_end = new_start + duration
        next_start = next(
            (new_starts[j] for j in range(i + 1, len(new_starts)) if new_starts[j] is not None),
            None,
        )
        if next_start is not None and new_end > next_start:
            new_end = next_start

        result.append(SRTSegment(
            index=orig_seg.index,
            start=new_start,
            end=new_end,
            text=orig_seg.text,
        ))
    return result


def transcribe_and_align(
    media_path: str,
    output_srt_path: str,
    language_code: Optional[str] = None,
    model_size: str = "large-v3",
    max_chars: int = 0,
    log: Callable[[str], None] = print,
    progress: Callable[[int], None] = lambda _: None,
    confirm_overwrite: Callable[[str], bool] = lambda _: True,
) -> None:
    device = get_device()
    compute_type = get_compute_type(device)
    log(f"장치: {device.upper()}")

    log("오디오 추출 중...")
    progress(10)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        extract_audio(media_path, tmp_wav)
        log("오디오 추출 완료.")
        progress(20)

        audio = whisperx.load_audio(tmp_wav)

        log(f"Whisper 모델 로드 중 ({model_size})...")
        progress(30)
        model = whisperx.load_model(
            model_size, device, compute_type=compute_type, language=language_code,
        )
        log("전사 중...")
        progress(40)
        result = model.transcribe(audio, batch_size=16)
        detected_lang = result.get("language", language_code or "en")
        log(f"전사 완료. 언어: {detected_lang}")
        progress(65)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        log(f"Alignment 모델 로드 중 (언어: {detected_lang})...")
        progress(70)
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        log("자막 정렬 중...")
        progress(80)
        aligned = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        log("정렬 완료.")
        progress(90)

        out_segments = aligned["segments"]
        if max_chars > 0:
            log(f"긴 자막 분할 중 (최대 {max_chars}자)...")
            nlp = load_spacy_model(detected_lang, log)
            out_segments = split_long_segments(out_segments, max_chars, nlp)

        segments = _wx_segments_to_srt(out_segments)
        if not segments:
            raise RuntimeError("생성된 자막이 없습니다.")

        final_path = build_output_path(output_srt_path, detected_lang)
        if os.path.exists(final_path) and not confirm_overwrite(final_path):
            log("취소되었습니다.")
            return

        write_srt(segments, final_path)
        progress(100)
        log(f"저장 완료: {final_path}")

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def align_srt(
    media_path: str,
    srt_path: str,
    output_srt_path: str,
    language_code: Optional[str] = None,
    max_chars: int = 0,
    log: Callable[[str], None] = print,
    progress: Callable[[int], None] = lambda _: None,
    confirm_overwrite: Callable[[str], bool] = lambda _: True,
) -> None:
    device = get_device()
    log(f"장치: {device.upper()}")

    log("오디오 추출 중...")
    progress(10)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        extract_audio(media_path, tmp_wav)
        log("오디오 추출 완료.")
        progress(25)

        log("SRT 파싱 중...")
        srt_segments = parse_srt(srt_path)
        if not srt_segments:
            raise ValueError("SRT 파일에서 자막을 읽을 수 없습니다.")
        log(f"자막 {len(srt_segments)}개 세그먼트 로드 완료.")

        audio = whisperx.load_audio(tmp_wav)

        if language_code is None:
            log("언어 자동 감지 중...")
            progress(35)
            detect_model = whisperx.load_model("small", device, compute_type="float32")
            detected = detect_model.transcribe(audio, batch_size=4)
            language_code = detected.get("language", "en")
            log(f"감지된 언어: {language_code}")
            del detect_model
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            log(f"선택된 언어: {language_code}")

        log(f"Alignment 모델 로드 중 (언어: {language_code})...")
        progress(50)
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        log("Alignment 모델 로드 완료.")

        log("자막 재정렬 중...")
        progress(70)
        wx_input = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in srt_segments
        ]
        aligned = whisperx.align(
            wx_input, model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        log("재정렬 완료.")
        progress(88)

        out_segments = aligned["segments"]
        if max_chars > 0:
            log(f"긴 자막 분할 중 (최대 {max_chars}자)...")
            nlp = load_spacy_model(language_code, log)
            out_segments = split_long_segments(out_segments, max_chars, nlp)
            new_segments = _wx_segments_to_srt(out_segments)
        else:
            log("원본 자막 길이 보존 적용 중...")
            new_segments = _merge_with_original_duration(out_segments, srt_segments)

        final_path = build_output_path(output_srt_path, language_code)
        if os.path.exists(final_path) and not confirm_overwrite(final_path):
            log("취소되었습니다.")
            return

        write_srt(new_segments, final_path)
        progress(100)
        log(f"저장 완료: {final_path}")

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
