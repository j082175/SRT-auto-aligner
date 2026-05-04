"""
자막 생성 및 재정렬 모듈.

ASR 엔진을 BaseEngine 인터페이스로 추상화 — 현재 구현: FasterWhisperEngine.
모드 1 - 생성+정렬: 영상 → engine.transcribe (faster-whisper word_timestamps) → SRT
모드 2 - 정렬만:   영상 + 기존 SRT → engine.align_to_srt (wav2vec2) → SRT

출력 파일명: {기본경로}.{언어코드}.srt  (예: movie.ko.srt)
"""

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, List, Optional

import ffmpeg
import requests
import torch
import unicodedata
import whisperx
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

# 마침표가 붙어도 문장 종결이 아닌 약어 (소문자, .!?, 콤마 떼고 비교)
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr",
    "vs", "etc", "ltd", "inc", "co", "corp",
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


def _is_sentence_break(word_text: str, next_word_text: str) -> bool:
    """word_text가 문장 종결 + next_word_text가 새 문장 시작인지.

    - 단어가 .!? 로 끝나야 함
    - 마침표는 약어(_ABBREVIATIONS)나 단일 글자(initial) 제외
    - 다음 단어의 첫 알파벳이 대문자여야 (따옴표/괄호는 건너뜀)
    """
    s = word_text.strip()
    if not s or s[-1] not in ".!?":
        return False
    if s[-1] == "." :
        bare = s.rstrip(".!?,").lower()
        if not bare or len(bare) <= 1 or bare in _ABBREVIATIONS:
            return False
    if not next_word_text:
        return False
    for c in next_word_text.lstrip():
        if c.isalpha():
            return c.isupper()
        if c in "\"'“‘(":
            continue
        return False
    return False


def _count_internal_sentence_breaks(words: List[dict]) -> int:
    """words 안에서 (마지막 제외) 실제 문장 boundary 개수."""
    count = 0
    for i in range(len(words) - 1):
        if _is_sentence_break(words[i].get("word", ""), words[i + 1].get("word", "")):
            count += 1
    return count


def _split_at_sentences(
    words: List[dict], seg_start: float, seg_end: float,
) -> List[dict]:
    """words를 _is_sentence_break 위치에서 분할. 약어로 인한 false positive 회피."""
    if not words:
        return []

    chunks_words: List[List[dict]] = []
    current: List[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        next_text = words[i + 1].get("word", "") if i + 1 < len(words) else ""
        if next_text and _is_sentence_break(w.get("word", ""), next_text):
            chunks_words.append(current)
            current = []
    if current:
        chunks_words.append(current)

    result: List[dict] = []
    for chunk_words in chunks_words:
        chunk_text = " ".join(w.get("word", "").strip() for w in chunk_words)
        result.append({
            "start": _first_timestamp(chunk_words, "start", seg_start),
            "end": _last_timestamp(chunk_words, "end", seg_end),
            "text": chunk_text,
            "words": chunk_words,
        })

    # 중간 청크 end = 다음 청크 start (끊김 방지), 마지막은 seg_end로
    for i in range(len(result) - 1):
        result[i]["end"] = result[i + 1]["start"]
    if result:
        result[-1]["end"] = seg_end

    return result


def split_long_segments(
    aligned_segments: List[dict],
    max_chars: int = 42,
    nlp: Optional[Any] = None,
) -> List[dict]:
    """
    긴 세그먼트 + 다중 문장 세그먼트 분할.
    - 글자 수 > max_chars : word-smart 분할 (명사구 경계, dangling 보정)
    - 문장 종결부호 2개 이상(약어 제외) : 짧아도 문장 단위로 분할
    - 둘 다 해당하면: 1) 문장 단위 사전 분할 → 2) 그래도 max_chars 초과한 sub만 word-smart 추가 분할
    word 타임스탬프 없으면 문장 부호 기준 + 글자 수 비례 시간 배분.
    """
    result = []
    for seg in aligned_segments:
        text = seg.get("text", "").strip()
        seg_start = seg.get("start")
        seg_end = seg.get("end")

        if seg_start is None or seg_end is None:
            result.append(seg)
            continue

        words = seg.get("words", [])
        has_timestamps = any(w.get("start") is not None for w in words)

        too_long = len(text) > max_chars
        if words and has_timestamps:
            multi_sentence = _count_internal_sentence_breaks(words) >= 1
        else:
            # 텍스트만 있을 때는 약어 무시한 거친 카운트
            multi_sentence = (text.count(".") + text.count("!") + text.count("?")) >= 2

        if not too_long and not multi_sentence:
            result.append(seg)
            continue

        if words and has_timestamps:
            # 1단계: 다중 문장이면 문장 단위 사전 분할
            if multi_sentence:
                pre_chunks = _split_at_sentences(words, seg_start, seg_end)
            else:
                pre_chunks = [{"start": seg_start, "end": seg_end, "text": text, "words": words}]

            # 2단계: 각 sub-chunk가 max_chars 초과하면 word-smart로 추가 분할
            final_chunks: List[dict] = []
            for pc in pre_chunks:
                pc_text = pc.get("text", "")
                pc_words = pc.get("words", [])
                if len(pc_text) > max_chars and pc_words:
                    joined = " ".join(w.get("word", "").strip() for w in pc_words)
                    noun_spans = _get_noun_chunk_spans(joined, nlp) if nlp else []
                    sub_chunks = _split_words_smart(
                        pc_words, pc["start"], pc["end"], max_chars, noun_spans,
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(pc)

            result.extend(_fix_dangling(final_chunks) if final_chunks else [seg])

        else:
            # word 타임스탬프 없음 → 문장 부호 기준 + 비례 분할
            if multi_sentence:
                # 다중 문장: 문장 종결부호로만 split (콤마 무시)
                parts = re.split(r'(?<=[.!?])\s+', text)
                chunks_text = [p.strip() for p in parts if p.strip()]
            else:
                # 단일 문장 too_long: 콤마 + 종결부호 기준 + 글자 수 누적
                parts = re.split(r'(?<=[.!?,])\s+', text)
                chunks_text = []
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


def _trim_silence_stretch(
    segments: List[dict],
    threshold: float = 3.0,
    buffer: float = 0.5,
) -> List[dict]:
    """segment 양 끝의 침묵 구간이 threshold 초 초과면 word 경계에 buffer만 남기고 트림.

    Whisper가 한 발화 안의 드라마틱 포즈를 segment에 포함시키는 경우 wav2vec2 word
    timestamp 기준으로 잘라낸다. threshold 미만의 자연스러운 침묵·호흡은 그대로 둔다.
    word timestamps가 없거나 None이면 원본 유지 (안전장치).

    한계: wav2vec2가 침묵 구간에 단어를 stretch하는 경우(Pyannote VAD가 드라마틱
    포즈를 segment에 포함시켜 결과 audio가 길어진 경우) word timestamp 자체가
    부정확해 이 트림이 못 잡는다. 그 케이스는 _trim_outlier_segments가 보완.
    """
    result = []
    for seg in segments:
        new_seg = dict(seg)
        words = seg.get("words", [])
        seg_start = seg.get("start")
        seg_end = seg.get("end")

        if not words or seg_start is None or seg_end is None:
            result.append(new_seg)
            continue

        first_word_start = _first_timestamp(words, "start", None)
        last_word_end = _last_timestamp(words, "end", None)

        if first_word_start is not None and (first_word_start - seg_start) > threshold:
            new_seg["start"] = first_word_start - buffer
        if last_word_end is not None and (seg_end - last_word_end) > threshold:
            new_seg["end"] = last_word_end + buffer

        result.append(new_seg)
    return result


def _trim_outlier_segments(
    segments: List[dict],
    outlier_factor: float = 3.0,
    min_threshold: float = 8.0,
    target_min: float = 5.0,
) -> List[dict]:
    """segment 길이가 다른 segment 대비 outlier로 길면 end-anchored로 트림.

    word timestamp가 부정확해 _trim_silence_stretch가 잡지 못하는 케이스를 보완.
    전체 segment의 median 길이 대비 outlier_factor 배 이상 + 최소 min_threshold초
    초과 segment만 발동 — 정상 영상에선 거의 작동하지 않는다.

    트림 후 길이는 max(median*outlier_factor, target_min)초. end는 그대로 두고
    start만 늦춰 leading silence(드라마틱 포즈)를 제거한다.
    """
    if len(segments) < 5:
        return segments  # 표본 너무 적으면 median 신뢰 못 함

    durations = []
    for seg in segments:
        s, e = seg.get("start"), seg.get("end")
        if s is not None and e is not None:
            durations.append(e - s)
    if not durations:
        return segments

    durations.sort()
    median = durations[len(durations) // 2]
    threshold = max(median * outlier_factor, min_threshold)
    target = max(median * outlier_factor, target_min)

    result = []
    for seg in segments:
        new_seg = dict(seg)
        s, e = seg.get("start"), seg.get("end")
        if s is None or e is None:
            result.append(new_seg)
            continue
        if (e - s) > threshold:
            new_seg["start"] = e - target
        result.append(new_seg)
    return result


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


# ============================================================
# Engine 추상화
# ============================================================


@dataclass
class EngineResult:
    """엔진의 전사·정렬 결과.

    segments: WhisperX 호환 dict — {start, end, text, words: [{word, start, end, ...}]}
    detected_language: 입력받았거나 엔진이 감지한 언어 코드 (예: "en")
    replacements_list: align_to_srt에서 wav2vec2 정확도 향상용 숫자→단어 트릭 산물.
        transcribe에선 빈 리스트, mode 2에선 세그먼트별 (단어, 원본숫자) 리스트.
    """
    segments: List[dict]
    detected_language: str
    replacements_list: list = field(default_factory=list)


class BaseEngine(ABC):
    """ASR 엔진 인터페이스 — 전사+alignment 또는 기존 SRT 재정렬 제공."""
    name: ClassVar[str] = "base"
    # 기존 SRT 재정렬(mode 2) 지원 여부. UI 측에서 선택 차단에 사용.
    supports_align_to_srt: ClassVar[bool] = True

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language_code: Optional[str],
        log: Callable[[str], None],
        progress: Callable[[int], None],
    ) -> EngineResult:
        """오디오 → 전사 + word-level 타임스탬프. progress 30~90 보고."""

    @abstractmethod
    def align_to_srt(
        self,
        audio_path: str,
        srt_segments: List[SRTSegment],
        language_code: Optional[str],
        log: Callable[[str], None],
        progress: Callable[[int], None],
    ) -> EngineResult:
        """기존 SRT 텍스트를 오디오 타이밍에 재정렬. progress 35~88 보고."""


class FasterWhisperEngine(BaseEngine):
    """Mode 1·2 모두 동일한 흐름:
    - whisperx.load_model (Pyannote VAD chunker + faster-whisper backend) 로 전사 —
      30초 단위 큰 chunk로 처리하므로 모델이 충분한 context를 받아 punctuation·대소문자·
      문장 구조가 살아 있는 transcription을 만든다.
    - wav2vec2 forced alignment 로 word 타임스탬프 정확화

    트레이드오프: WhisperX의 batched 처리는 침묵·불분명 구간에서 영상 후반부 대사가
    끼어드는 환각 가능성이 있다. 다만 faster-whisper 직접 방식은 punctuation을 못 만들어
    OCR 오류처럼 읽혀, 사용자 평가상 환각이 있더라도 WhisperX 쪽이 훨씬 나음.
    """
    name: ClassVar[str] = "fasterwhisper"

    def __init__(self, model_size: str = "large-v3"):
        self.model_size = model_size

    def transcribe(self, audio_path, language_code, log, progress):
        device = get_device()
        compute_type = get_compute_type(device)

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000

        log(f"Whisper 모델 로드 중 ({self.model_size})...")
        progress(30)
        model = whisperx.load_model(
            self.model_size, device, compute_type=compute_type,
            language=language_code,
            vad_options={"vad_onset": 0.3, "vad_offset": 0.2},
        )
        log("전사 중...")
        progress(40)

        # 전사 중 진행 바를 40→65% 구간에서 시간 기반으로 천천히 전진
        _done = threading.Event()
        def _advance():
            t0 = time.time()
            while not _done.is_set():
                frac = min((time.time() - t0) / max(duration, 1), 0.95)
                progress(40 + frac * 25)
                time.sleep(0.3)
        threading.Thread(target=_advance, daemon=True).start()

        result = model.transcribe(audio, batch_size=16)
        _done.set()

        detected_lang = result.get("language", language_code or "en")
        segments = result.get("segments", [])
        log(f"전사 완료. 언어: {detected_lang} | 세그먼트: {len(segments)}개")
        progress(65)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        if not segments:
            return EngineResult(
                segments=[],
                detected_language=detected_lang,
                replacements_list=[],
            )

        log(f"Alignment 모델 로드 중 (언어: {detected_lang})...")
        progress(70)
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device,
        )

        log(f"자막 정렬 중... ({len(segments)}개 세그먼트)")
        aligned = _align_segments_with_progress(
            segments, model_a, metadata, audio, device,
            80, 90, progress, log,
        )

        # 드라마틱 포즈로 segment가 비정상적으로 길어진 케이스 정리 (16초짜리 한 줄 등).
        # 1) word timestamp 기준: 단어 단위 침묵이 3초 초과면 트림 (정확)
        # 2) outlier 길이 기준: median 대비 3배 + 8초 초과 segment를 end-anchored 트림
        #    (wav2vec2가 침묵에 단어를 stretch한 경우 보완)
        # 두 트림 모두 임계값이 보수적이라 정상 segment(0.5~5초)는 영향 안 받음.
        out_segments = _trim_silence_stretch(aligned["segments"])
        out_segments = _trim_outlier_segments(out_segments)

        log("Alignment 완료.")
        progress(90)

        return EngineResult(
            segments=out_segments,
            detected_language=detected_lang,
            replacements_list=[],
        )

    def align_to_srt(self, audio_path, srt_segments, language_code, log, progress):
        device = get_device()
        audio = whisperx.load_audio(audio_path)

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

        return EngineResult(
            segments=aligned["segments"],
            detected_language=language_code,
            replacements_list=replacements_list,
        )


# ------------------------------------------------------------
# Qwen3 결과 → WhisperX 호환 segment 변환
# ------------------------------------------------------------

def _normalize_token(token: str) -> str:
    """Qwen3 단어 매칭 비교용 — 알파벳·숫자·한글만 남기고 소문자."""
    return re.sub(r"[^a-z0-9가-힣]", "", token.lower())


def _convert_qwen3_to_segments(text: str, items: List[dict]) -> List[dict]:
    """Qwen3 출력(전체 텍스트 + 단어 타임스탬프)을 WhisperX 호환 segment 리스트로 변환.

    구두점은 sentence 분할에 사용하고 결과 텍스트·words에 보존. items는 단순 단어이므로
    normalize 매칭으로 sentence 단어와 정렬 (qwen3_eval/json_to_srt_sentences.py 로직 차용).
    """
    if not items:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]

    word_idx = 0
    segments: List[dict] = []
    for sentence in sentences:
        sent_words = sentence.split()
        if not sent_words:
            continue

        matched: List[dict] = []
        for sw in sent_words:
            sw_norm = _normalize_token(sw)
            if not sw_norm:
                continue
            search_limit = min(word_idx + 5, len(items))
            for j in range(word_idx, search_limit):
                if _normalize_token(items[j]["text"]) == sw_norm:
                    matched.append({
                        "word": sw,  # 원본 (구두점 포함)
                        "start": items[j]["start"],
                        "end": items[j]["end"],
                    })
                    word_idx = j + 1
                    break

        if not matched:
            continue

        segments.append({
            "start": matched[0]["start"],
            "end": matched[-1]["end"],
            "text": sentence,
            "words": matched,
        })

    return segments


class Qwen3Engine(BaseEngine):
    """Qwen3-ASR + ForcedAligner-0.6B (별도 venv subprocess).

    동작 조건:
    - Qwen3 venv python 위치: QWEN3_VENV_PYTHON 환경변수 > <project>/qwen3_venv/.venv > 개발용 fallback
    - 오디오 경로 ASCII (nagisa C 확장이 한국어 경로 미지원)
    - mode 2 (기존 SRT 재정렬)는 ForcedAligner의 silence drift 약점으로 미지원
    """
    name: ClassVar[str] = "qwen3"
    supports_align_to_srt: ClassVar[bool] = False

    _DEV_FALLBACK_PYTHON = r"C:\Users\j0821\qwen3_eval\.venv\Scripts\python.exe"

    def __init__(self, model: str = "0.6B"):
        if model not in ("0.6B", "1.7B"):
            raise ValueError(f"Qwen3 모델은 0.6B 또는 1.7B만 가능: {model}")
        self.model = model

    @classmethod
    def find_python(cls) -> Optional[str]:
        env = os.environ.get("QWEN3_VENV_PYTHON")
        if env and os.path.isfile(env):
            return env
        here = os.path.dirname(os.path.abspath(__file__))
        sibling = os.path.join(here, "qwen3_venv", ".venv", "Scripts", "python.exe")
        if os.path.isfile(sibling):
            return sibling
        if os.path.isfile(cls._DEV_FALLBACK_PYTHON):
            return cls._DEV_FALLBACK_PYTHON
        return None

    def transcribe(self, audio_path, language_code, log, progress):
        py = self.find_python()
        if py is None:
            raise RuntimeError(
                "Qwen3 venv python을 찾을 수 없습니다. QWEN3_VENV_PYTHON 환경변수를 설정하거나 "
                "<프로젝트>/qwen3_venv/.venv 에 설치하세요."
            )
        try:
            audio_path.encode("ascii")
        except UnicodeEncodeError:
            raise RuntimeError(
                f"Qwen3는 ASCII 경로만 지원합니다 (받은 경로: {audio_path}). "
                "한글이 포함된 환경에서는 FasterWhisper를 사용하세요."
            )

        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_runner.py")
        cmd = [py, runner, "--audio", audio_path, "--model", self.model]
        if language_code:
            cmd.extend(["--language", language_code])

        log(f"Qwen3 subprocess 시작 ({self.model})...")
        progress(30)

        # 자식 stdout/stderr를 UTF-8로 강제 — Windows 기본 cp949 인코딩과 부모 디코더 mismatch 방지
        # (mismatch 시 한글 [STAGE] 로그와 한국어 전사 결과가 mojibake가 됨)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        stderr_tail: List[str] = []

        def pump_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                if line.startswith("[STAGE] "):
                    log(line[len("[STAGE] "):])
                elif line.startswith("[PROGRESS] "):
                    try:
                        progress(int(line[len("[PROGRESS] "):]))
                    except ValueError:
                        pass
                else:
                    stderr_tail.append(line)
                    if len(stderr_tail) > 50:
                        del stderr_tail[0]

        t = threading.Thread(target=pump_stderr, daemon=True)
        t.start()

        stdout, _ = proc.communicate()
        t.join(timeout=2.0)

        # stdout 마지막 비어있지 않은 줄이 결과 JSON
        last_line = ""
        for line in reversed(stdout.splitlines()):
            if line.strip():
                last_line = line
                break

        if not last_line:
            tail = "\n".join(stderr_tail[-15:])
            raise RuntimeError(f"Qwen3 결과 누락 (exit {proc.returncode}):\n{tail}")

        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError as e:
            tail = "\n".join(stderr_tail[-15:])
            raise RuntimeError(f"Qwen3 JSON 파싱 실패: {e}\nlast: {last_line[:200]}\nstderr:\n{tail}")

        if not payload.get("ok"):
            raise RuntimeError(f"Qwen3 실패: {payload.get('error', 'unknown')}")
        if proc.returncode != 0:
            raise RuntimeError(f"Qwen3 subprocess exit {proc.returncode}")

        segments = _convert_qwen3_to_segments(payload["text"], payload["items"])
        if not segments:
            raise RuntimeError("Qwen3 결과에서 segment를 추출하지 못했습니다.")
        log(f"Qwen3 완료 — segment {len(segments)}개.")
        progress(90)

        return EngineResult(
            segments=segments,
            detected_language=payload["language_detected"],
            replacements_list=[],  # Qwen3는 숫자→단어 트릭 불필요
        )

    def align_to_srt(self, audio_path, srt_segments, language_code, log, progress):
        raise NotImplementedError(
            "Qwen3 엔진은 mode 2 (기존 SRT 재정렬)를 지원하지 않습니다. "
            "ForcedAligner의 silence drift로 wav2vec2보다 정확도가 낮기 때문입니다. "
            "기존 SRT 재정렬에는 FasterWhisper 엔진을 사용하세요."
        )


# ------------------------------------------------------------
# Together API (whisper-large-v3 클라우드 추론)
# ------------------------------------------------------------

def _convert_together_to_segments(data: dict) -> List[dict]:
    """Together verbose_json 응답(segments + words 분리)을 WhisperX 호환 segment로 변환.

    각 segment에 자기 시간 범위에 속하는 words를 모아 붙임. word.start < segment.end 기준.
    """
    segments = data.get("segments") or []
    words = list(data.get("words") or [])
    if not segments:
        return []

    word_idx = 0
    result: List[dict] = []
    for seg in segments:
        seg_end = seg.get("end")
        seg_words: List[dict] = []
        while word_idx < len(words) and words[word_idx].get("start", 0) < seg_end:
            w = words[word_idx]
            seg_words.append({
                "word": w.get("word", ""),
                "start": w.get("start"),
                "end": w.get("end"),
            })
            word_idx += 1
        result.append({
            "start": seg.get("start"),
            "end": seg_end,
            "text": (seg.get("text") or "").strip(),
            "words": seg_words,
        })
    return result


class TogetherEngine(BaseEngine):
    """Together API의 whisper-large-v3 클라우드 추론.

    - TOGETHER_API_KEY 환경변수 필요
    - 1시간 영상 기준 1.5~2분 (로컬 faster-whisper 대비 약 10배 빠름)
    - mode 2 (기존 SRT 재정렬) 미지원 — API에 alignment 엔드포인트 없음
    - 긴 영상은 자동으로 N분 단위 mp3 청크로 분할해 순차 업로드 후
      segment 타임스탬프에 청크 오프셋을 더해 합침 (단일 업로드 한도 회피).
    """
    name: ClassVar[str] = "together"
    supports_align_to_srt: ClassVar[bool] = False

    API_URL: ClassVar[str] = "https://api.together.xyz/v1/audio/transcriptions"
    CHUNK_SECONDS: ClassVar[int] = 1200       # 청크 길이 (20분)
    CHUNK_BITRATE: ClassVar[str] = "64k"      # mp3 비트레이트 (1분 ≈ 480KB)

    def __init__(self, model: str = "openai/whisper-large-v3"):
        self.model = model

    def transcribe(self, audio_path, language_code, log, progress):
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY 환경변수가 없습니다. "
                "https://api.together.ai/settings/api-keys 에서 발급받아 설정하세요."
            )

        # 1) 오디오를 N분 단위 mp3 청크로 분할 (단일 업로드 한도 회피)
        chunks_dir = tempfile.mkdtemp(prefix="together_chunks_")
        try:
            log(f"오디오를 {self.CHUNK_SECONDS // 60}분 청크로 분할 중...")
            progress(15)
            chunk_paths = self._split_to_mp3_chunks(audio_path, chunks_dir)
            if not chunk_paths:
                raise RuntimeError("청크 분할 결과가 비어있습니다.")
            log(f"청크 {len(chunk_paths)}개 생성. Together API로 순차 업로드합니다.")

            # 2) 청크별 transcribe → 절대 시간으로 보정해 누적
            all_segments: List[dict] = []
            all_words: List[dict] = []
            detected_lang: Optional[str] = None
            base = 20.0
            span = 70.0  # 20% → 90% 사이에서 청크별 균등 분배

            for i, chunk in enumerate(chunk_paths):
                chunk_offset = i * self.CHUNK_SECONDS
                size_mb = os.path.getsize(chunk) / 1_048_576
                log(
                    f"청크 {i+1}/{len(chunk_paths)} 업로드 중 "
                    f"(오프셋 {chunk_offset // 60}분, {size_mb:.1f}MB)..."
                )
                progress(int(base + span * i / len(chunk_paths)))

                data = self._transcribe_one(chunk, api_key, language_code)
                if detected_lang is None:
                    detected_lang = data.get("language") or language_code or "en"

                for seg in (data.get("segments") or []):
                    seg = dict(seg)
                    if seg.get("start") is not None:
                        seg["start"] = seg["start"] + chunk_offset
                    if seg.get("end") is not None:
                        seg["end"] = seg["end"] + chunk_offset
                    all_segments.append(seg)
                for w in (data.get("words") or []):
                    w = dict(w)
                    if w.get("start") is not None:
                        w["start"] = w["start"] + chunk_offset
                    if w.get("end") is not None:
                        w["end"] = w["end"] + chunk_offset
                    all_words.append(w)

            progress(90)
            data = {
                "language": detected_lang,
                "segments": all_segments,
                "words": all_words,
            }
        finally:
            # 청크 임시 파일 정리
            try:
                for name in os.listdir(chunks_dir):
                    try:
                        os.remove(os.path.join(chunks_dir, name))
                    except OSError:
                        pass
                os.rmdir(chunks_dir)
            except OSError:
                pass

        detected_lang = data.get("language") or language_code or "en"

        segments = _convert_together_to_segments(data)
        if not segments:
            raise RuntimeError("Together API 응답에서 segment를 추출하지 못했습니다.")

        log(
            f"전사 완료. 언어: {detected_lang} | "
            f"segment {len(segments)}개, word {len(data.get('words') or [])}개"
        )
        progress(95)

        return EngineResult(
            segments=segments,
            detected_language=detected_lang,
            replacements_list=[],
        )

    def _split_to_mp3_chunks(self, audio_path: str, out_dir: str) -> List[str]:
        """ffmpeg segment muxer로 N초 단위 mp3 청크 생성. 정렬된 청크 경로 리스트 반환."""
        pattern = os.path.join(out_dir, "chunk_%04d.mp3")
        (
            ffmpeg
            .input(audio_path)
            .output(
                pattern,
                format="segment",
                segment_time=self.CHUNK_SECONDS,
                ar=16000,
                ac=1,
                acodec="libmp3lame",
                audio_bitrate=self.CHUNK_BITRATE,
                reset_timestamps=1,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return sorted(
            os.path.join(out_dir, name)
            for name in os.listdir(out_dir)
            if name.startswith("chunk_") and name.endswith(".mp3")
        )

    def _transcribe_one(self, file_path: str, api_key: str, language_code) -> dict:
        """단일 mp3 청크를 Together API로 전사. verbose_json 응답 dict 반환."""
        form: List[tuple] = [
            ("model", self.model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
        if language_code:
            form.append(("language", language_code))

        with open(file_path, "rb") as f:
            resp = requests.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(file_path), f, "audio/mpeg")},
                data=form,
                timeout=600,
            )

        if resp.status_code != 200:
            body = resp.text[:500]
            raise RuntimeError(f"Together API 실패 (status {resp.status_code}): {body}")
        return resp.json()

    def align_to_srt(self, audio_path, srt_segments, language_code, log, progress):
        raise NotImplementedError(
            "Together 엔진은 mode 2 (기존 SRT 재정렬)를 지원하지 않습니다. "
            "API에 alignment 엔드포인트가 없어 새 전사만 가능합니다. "
            "기존 SRT 재정렬에는 FasterWhisper 엔진을 사용하세요."
        )


# ------------------------------------------------------------
# 엔진 팩토리
# ------------------------------------------------------------

ENGINES: List[str] = ["fasterwhisper", "qwen3", "together"]


def create_engine(name: str, **kwargs) -> BaseEngine:
    """이름으로 엔진 인스턴스 생성. kwargs는 엔진별 옵션.

    fasterwhisper: model_size (str)
    qwen3:         qwen3_model (str: "0.6B" | "1.7B")
    together:      together_model (str, 기본 "openai/whisper-large-v3")
    """
    name = (name or "").lower()
    if name == "fasterwhisper":
        return FasterWhisperEngine(model_size=kwargs.get("model_size", "large-v3"))
    if name == "qwen3":
        return Qwen3Engine(model=kwargs.get("qwen3_model", "0.6B"))
    if name == "together":
        return TogetherEngine(model=kwargs.get("together_model", "openai/whisper-large-v3"))
    raise ValueError(f"지원되지 않는 엔진: {name} (사용 가능: {ENGINES})")


# ============================================================
# 호스트 함수 — 오디오 추출, 엔진 위임, split·post-process, write
# ============================================================


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
    engine: Optional[BaseEngine] = None,
) -> None:
    if engine is None:
        engine = FasterWhisperEngine(model_size=model_size)

    device = get_device()
    log(f"장치: {device.upper()}")

    log("오디오 추출 중...")
    progress(10)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        extract_audio(media_path, tmp_wav)
        log("오디오 추출 완료.")
        progress(20)

        result = engine.transcribe(tmp_wav, language_code, log, progress)
        detected_lang = result.detected_language

        out_segments = result.segments
        if max_chars > 0:
            log(f"긴 자막 분할 중 (최대 {max_chars}자)...")
            nlp = load_spacy_model(detected_lang, log)
            out_segments = split_long_segments(out_segments, max_chars, nlp)

        segments = _wx_segments_to_srt(out_segments)
        segments = collapse_numbers_in_srt(segments, result.replacements_list)
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
    engine: Optional[BaseEngine] = None,
) -> None:
    if engine is None:
        engine = FasterWhisperEngine()

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

        result = engine.align_to_srt(tmp_wav, srt_segments, language_code, log, progress)
        detected_lang = result.detected_language

        out_segments = result.segments
        if max_chars > 0:
            log(f"긴 자막 분할 중 (최대 {max_chars}자)...")
            nlp = load_spacy_model(detected_lang, log)
            out_segments = split_long_segments(out_segments, max_chars, nlp)
            new_segments = _wx_segments_to_srt(out_segments)
        else:
            log("원본 자막 길이 보존 적용 중...")
            new_segments = _merge_with_original_duration(out_segments, srt_segments)

        new_segments = collapse_numbers_in_srt(new_segments, result.replacements_list)

        final_path = build_output_path(output_folder, media_path, detected_lang)
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
