"""
WhisperX 기반 자막 생성 및 재정렬 모듈.

모드 1 - 생성+정렬: 영상 → Whisper 전사 → wav2vec2 alignment → SRT
모드 2 - 정렬만:   영상 + 기존 SRT → wav2vec2 alignment → SRT

출력 파일명: {기본경로}_{언어코드}.srt  (예: movie_ko.srt)
"""

import os
import tempfile
from typing import Callable, List, Optional

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


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "float32"


def build_output_path(base_srt_path: str, lang_code: str) -> str:
    """'{경로}/{파일명}.srt' → '{경로}/{파일명}_{lang}.srt'"""
    root, ext = os.path.splitext(base_srt_path)
    return f"{root}_{lang_code}{ext}"


def extract_audio(input_path: str, output_wav: str) -> None:
    """영상/오디오 파일에서 16kHz mono wav 추출."""
    (
        ffmpeg
        .input(input_path)
        .output(output_wav, ar=16000, ac=1, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


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
    """새 start + 원본 duration → new_end. 다음 세그먼트 start 초과 시 클리핑."""
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
    log: Callable[[str], None] = print,
    progress: Callable[[int], None] = lambda _: None,
    confirm_overwrite: Callable[[str], bool] = lambda _: True,
) -> None:
    """
    모드 1: 자막 생성 + 정렬을 한 번에 처리.
    출력: {output_srt_path 기본명}_{언어코드}.srt
    """
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
        progress(95)

        segments = _wx_segments_to_srt(aligned["segments"])
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
    log: Callable[[str], None] = print,
    progress: Callable[[int], None] = lambda _: None,
    confirm_overwrite: Callable[[str], bool] = lambda _: True,
) -> None:
    """
    모드 2: 기존 SRT의 타임스탬프만 재정렬.
    출력: {output_srt_path 기본명}_{언어코드}.srt
    """
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
        progress(90)

        log("원본 자막 길이 보존 적용 중...")
        new_segments = _merge_with_original_duration(aligned["segments"], srt_segments)

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
