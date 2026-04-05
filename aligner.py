"""
WhisperX 기반 자막 생성 및 재정렬 모듈.

모드 1 - 생성+정렬: 영상 → Whisper 전사 → wav2vec2 alignment → SRT
모드 2 - 정렬만:   영상 + 기존 SRT → wav2vec2 alignment → SRT

출력 파일명: {기본경로}.{언어코드}.srt  (예: movie.ko.srt)
"""

import os
import re
import tempfile
import threading
import time
from typing import Any, Callable, List, Optional

import ffmpeg
import torch
import unicodedata
import whisperx
from faster_whisper import WhisperModel
from num2words import num2words

from srt_utils import SRTSegment, parse_srt, write_srt, write_txt


# 언어 코드 → num2words 언어 코드 매핑
_NUM2WORDS_LANG = {
    "en": "en", "ko": "ko", "ja": "ja", "zh": "zh",
    "es": "es", "fr": "fr", "de": "de", "ru": "ru",
    "pt": "pt", "it": "it", "ar": "ar",
}


def _expand_numbers(text: str, lang: str) -> tuple:
    """텍스트 내 숫자를 해당 언어의 단어로 변환.

    Returns:
        (expanded_text, replacements) — replacements는 (원본숫자, 변환단어) 리스트
    """
    n2w_lang = _NUM2WORDS_LANG.get(lang, "en")
    replacements = []

    def replace(m):
        num_str = m.group(0)
        try:
            n = int(num_str)
            word = num2words(n, lang=n2w_lang)
            replacements.append((word, num_str))
            return word
        except Exception:
            return num_str

    expanded = re.sub(r"\b\d+\b", replace, text)
    return expanded, replacements


def _collapse_numbers(text: str, replacements: list) -> str:
    """_expand_numbers의 역변환: 변환된 단어(하이픈/공백 모두)를 원래 숫자로 되돌림."""
    for word, orig in replacements:
        # "twenty-two" 또는 "twenty two" 두 형태 모두 매칭
        pattern = re.escape(word).replace(r"\-", r"[-\s]").replace(r"\ ", r"[-\s]")
        text = re.sub(pattern, orig, text, flags=re.IGNORECASE)
    return text


def _restore_segments(aligned_segments: list, orig_texts: list, replacements_list: list) -> None:
    """alignment 결과의 seg["text"]와 word["word"]를 원본으로 복원."""
    for seg, orig, replacements in zip(aligned_segments, orig_texts, replacements_list):
        seg["text"] = orig
        for w in seg.get("words", []):
            w["word"] = _collapse_numbers(w.get("word", ""), replacements)


def collapse_numbers_in_srt(segments: List[SRTSegment], replacements_list: list) -> List[SRTSegment]:
    """최종 SRT 세그먼트의 텍스트에서 숫자 단어를 원래 숫자로 복원 (split 후 잔여 처리)."""
    # 모든 replacements를 하나로 합침
    all_replacements = [r for reps in replacements_list for r in reps]
    if not all_replacements:
        return segments
    for seg in segments:
        seg.text = _collapse_numbers(seg.text, all_replacements)
    return segments

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
    return "int8_float16" if device == "cuda" else "int8"


def build_output_path(output_folder: str, input_path: str, lang_code: str) -> str:
    """출력 폴더 + 입력 파일명 + 언어코드로 최종 SRT 경로 생성.

    예: ('/Videos', '/Movies/movie.mp4', 'en') → '/Videos/movie.en.srt'
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_folder, f"{basename}.{lang_code}.srt")


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


def _first_timestamp(words: List[dict], key: str, fallback: float) -> float:
    """words 중 key 타임스탬프가 있는 첫 번째 값 반환."""
    for w in words:
        v = w.get(key)
        if v is not None:
            return v
    return fallback


def _last_timestamp(words: List[dict], key: str, fallback: float) -> float:
    """words 중 key 타임스탬프가 있는 마지막 값 반환."""
    for w in reversed(words):
        v = w.get(key)
        if v is not None:
            return v
    return fallback


def _split_words_smart(
    words: List[dict],
    seg_start: float,
    seg_end: float,
    max_chars: int,
    noun_spans: List[tuple],
) -> List[dict]:
    """
    word 타임스탬프 기준 분할.
    - 타임스탬프 없는 단어도 텍스트에 포함 (누락 방지)
    - 명사구 중간에서 자르지 않도록 cut point 조정
    - 중간 청크 end = 다음 청크 start (끊김 방지)
    """
    word_texts = [w.get("word", "").strip() for w in words]

    # 각 단어의 문자 오프셋 계산
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
        remaining = " ".join(word_texts[start_idx:])
        if len(remaining) <= max_chars:
            chunk_words = words[start_idx:]
            chunks.append({
                "start": _first_timestamp(chunk_words, "start", seg_start),
                "end": seg_end,
                "text": remaining,
                "words": chunk_words,
            })
            break

        # max_chars 초과 첫 번째 단어 인덱스
        overflow_idx = start_idx
        for i in range(start_idx, len(words)):
            chunk_len = char_ends[i] - char_starts[start_idx]
            if chunk_len > max_chars:
                overflow_idx = i
                break
        else:
            overflow_idx = len(words)

        # 1순위: 역방향으로 콤마 위치 탐색 (앞 청크가 최소 25자 이상일 때만)
        cut_idx = None
        for i in range(overflow_idx - 1, start_idx, -1):
            if word_texts[i].rstrip().endswith(","):
                chunk_len = char_ends[i] - char_starts[start_idx]
                if chunk_len >= 25:
                    cut_idx = i
                    break

        # 2순위: 역방향으로 spaCy 명사구 경계 탐색
        if cut_idx is None:
            for i in range(overflow_idx - 1, start_idx, -1):
                if _is_safe_cut(char_ends[i], noun_spans):
                    cut_idx = i
                    break

        # 3순위: 정방향 탐색
        if cut_idx is None:
            for i in range(overflow_idx, len(words)):
                if _is_safe_cut(char_ends[i], noun_spans):
                    cut_idx = i
                    break

        if cut_idx is None:
            cut_idx = max(overflow_idx - 1, start_idx)

        chunk_words = words[start_idx: cut_idx + 1]
        chunks.append({
            "start": _first_timestamp(chunk_words, "start", seg_start),
            "end": _last_timestamp(chunk_words, "end", seg_end),  # 임시, 아래서 보정
            "text": " ".join(word_texts[start_idx: cut_idx + 1]),
            "words": chunk_words,
        })
        start_idx = cut_idx + 1

    # 중간 청크 end = 다음 청크 start (빈 구간 없애기)
    for i in range(len(chunks) - 1):
        chunks[i]["end"] = chunks[i + 1]["start"]

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

        # 타임스탬프 없는 단어도 포함 (텍스트 누락 방지)
        words = seg.get("words", [])
        # 분할 기준으로 쓸 타임스탬프가 하나라도 있으면 word 기반 분할
        has_timestamps = any(w.get("start") is not None for w in words)

        if words and has_timestamps:
            joined = " ".join(w.get("word", "").strip() for w in words)
            noun_spans = _get_noun_chunk_spans(joined, nlp) if nlp else []

            chunks = _split_words_smart(words, seg_start, seg_end, max_chars, noun_spans)
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


def _fmt_time(s: float) -> str:
    """초 → MM:SS.mmm 포맷."""
    m = int(s // 60)
    return f"{m:02d}:{s - m * 60:06.3f}"


def _align_segments_with_progress(
    segments: List[dict],
    model_a,
    metadata,
    audio,
    device: str,
    p_start: float,
    p_end: float,
    progress: Callable[[float], None],
    log: Callable[[str], None],
) -> dict:
    """전체 세그먼트를 한 번에 정렬하고 결과를 로그로 출력."""
    progress(p_start)
    r = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
    for seg in r.get("segments", []):
        t0 = _fmt_time(seg.get("start") or 0)
        t1 = _fmt_time(seg.get("end") or 0)
        log(f"[{t0} --> {t1}] {seg.get('text', '').strip()}")
    progress(p_end)
    return r


def _is_hallucination(text: str) -> bool:
    """음표·이모지만으로 구성된 hallucination 세그먼트 여부 판별."""
    stripped = text.strip()
    if not stripped:
        return True
    for ch in stripped:
        cat = unicodedata.category(ch)
        if cat not in ("So", "Cf", "Zs") and not ch.isspace():
            return False
    return True


def _wx_segments_to_srt(wx_segments: List[dict]) -> List[SRTSegment]:
    result = []
    for i, seg in enumerate(wx_segments, start=1):
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "").strip()
        if start is None or end is None or not text:
            continue
        if _is_hallucination(text):
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
    output_folder: str,
    language_code: Optional[str] = None,
    model_size: str = "large-v3",
    max_chars: int = 0,
    save_txt: bool = False,
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
        duration = len(audio) / 16000

        log(f"Whisper 모델 로드 중 ({model_size})...")
        progress(30)
        fw_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log("전사 중...")
        progress(40)

        # 전사 중 진행 바를 40→90% 구간에서 시간 기반으로 천천히 전진
        _done = threading.Event()
        def _advance():
            t0 = time.time()
            while not _done.is_set():
                frac = min((time.time() - t0) / max(duration, 1), 0.95)
                progress(40 + frac * 50)
                time.sleep(0.3)
        threading.Thread(target=_advance, daemon=True).start()

        segments_gen, info = fw_model.transcribe(
            tmp_wav,
            word_timestamps=True,
            language=language_code,
            beam_size=5,
            condition_on_previous_text=False,
        )
        result_segments = []
        for seg in segments_gen:
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "score": w.probability}
                for w in (seg.words or [])
            ]
            result_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
            })
        _done.set()

        detected_lang = info.language or language_code or "en"
        log(f"전사 완료. 언어: {detected_lang} | 세그먼트: {len(result_segments)}개")
        progress(90)

        out_segments = result_segments
        if max_chars > 0:
            log(f"긴 자막 분할 중 (최대 {max_chars}자)...")
            nlp = load_spacy_model(detected_lang, log)
            out_segments = split_long_segments(out_segments, max_chars, nlp)

        segments = _wx_segments_to_srt(out_segments)
        replacements_list = [[] for _ in segments]
        if not segments:
            raise RuntimeError("생성된 자막이 없습니다.")

        final_path = build_output_path(output_folder, media_path, detected_lang)
        # 자동 감지인 경우에만 후확인 (명시 선택 시 시작 전 이미 확인)
        if language_code is None and os.path.exists(final_path) and not confirm_overwrite(final_path):
            log("취소되었습니다.")
            return

        write_srt(segments, final_path)
        if save_txt:
            txt_path = os.path.splitext(final_path)[0] + ".txt"
            write_txt(segments, txt_path)
            log(f"저장 완료: {final_path}, {os.path.basename(txt_path)}")
        else:
            log(f"저장 완료: {final_path}")
        progress(100)

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def align_srt(
    media_path: str,
    srt_path: str,
    output_folder: str,
    language_code: Optional[str] = None,
    max_chars: int = 0,
    save_txt: bool = False,
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

        orig_texts = [seg.text for seg in srt_segments]
        replacements_list = []
        wx_input = []
        for seg in srt_segments:
            expanded, replacements = _expand_numbers(seg.text, language_code)
            wx_input.append({"start": seg.start, "end": seg.end, "text": expanded})
            replacements_list.append(replacements)

        log(f"자막 재정렬 중... ({len(wx_input)}개 세그먼트)")
        progress(70)
        aligned = _align_segments_with_progress(
            wx_input, model_a, metadata, audio, device,
            70, 88, progress, log,
        )
        _restore_segments(aligned["segments"], orig_texts, replacements_list)
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

        new_segments = collapse_numbers_in_srt(new_segments, replacements_list)

        final_path = build_output_path(output_folder, media_path, language_code)
        # 자동 감지인 경우에만 후확인 (명시 선택 시 시작 전 이미 확인)
        if language_code is None and os.path.exists(final_path) and not confirm_overwrite(final_path):
            log("취소되었습니다.")
            return

        write_srt(new_segments, final_path)
        if save_txt:
            txt_path = os.path.splitext(final_path)[0] + ".txt"
            write_txt(new_segments, txt_path)
            log(f"저장 완료: {final_path}, {os.path.basename(txt_path)}")
        else:
            log(f"저장 완료: {final_path}")
        progress(100)

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
