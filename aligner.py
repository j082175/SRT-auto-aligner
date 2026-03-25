"""
WhisperX 기반 자막 재정렬 모듈.

워크플로우:
1. 영상/오디오에서 오디오 추출
2. SRT 파싱 → 원본 duration 저장
3. whisperx.align() 로 새 start 시간 획득
4. new_end = new_start + 원본_duration 으로 재조합
5. 새 SRT 저장
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


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_audio(input_path: str, output_wav: str) -> None:
    """영상/오디오 파일에서 16kHz mono wav 추출."""
    (
        ffmpeg
        .input(input_path)
        .output(output_wav, ar=16000, ac=1, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


def _merge_with_original_duration(
    aligned_segments: List[dict],
    original_segments: List[SRTSegment],
) -> List[SRTSegment]:
    """
    WhisperX로 얻은 새 start에 원본 duration을 붙여 세그먼트 재조합.
    다음 세그먼트의 start가 현재 end보다 앞서면 현재 end를 잘라냄(클리핑).
    """
    # 1패스: new_start 수집
    new_starts: List[Optional[float]] = []
    aligned_idx = 0
    for orig_seg in original_segments:
        if aligned_idx < len(aligned_segments):
            new_starts.append(aligned_segments[aligned_idx].get("start"))
            aligned_idx += 1
        else:
            new_starts.append(None)

    # 2패스: new_end = min(new_start + 원본_duration, 다음_new_start)
    result = []
    for i, orig_seg in enumerate(original_segments):
        new_start = new_starts[i]
        duration = orig_seg.end - orig_seg.start

        if new_start is None:
            result.append(orig_seg)
            continue

        new_end = new_start + duration

        # 다음 세그먼트 start가 있으면 그 앞에서 끊기
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


def align_srt(
    media_path: str,
    srt_path: str,
    output_srt_path: str,
    language_code: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> None:
    device = get_device()
    log(f"장치: {device.upper()}")

    log("오디오 추출 중...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        extract_audio(media_path, tmp_wav)
        log("오디오 추출 완료.")

        log("SRT 파싱 중...")
        srt_segments = parse_srt(srt_path)
        if not srt_segments:
            raise ValueError("SRT 파일에서 자막을 읽을 수 없습니다.")
        log(f"자막 {len(srt_segments)}개 세그먼트 로드 완료.")

        audio = whisperx.load_audio(tmp_wav)

        if language_code is None:
            log("언어 자동 감지 중...")
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
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        log("Alignment 모델 로드 완료.")

        log("자막 재정렬 중...")
        wx_input = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in srt_segments
        ]
        aligned = whisperx.align(
            wx_input,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        log("재정렬 완료.")

        # 새 start + 원본 duration → new_end
        log("원본 자막 길이 보존 적용 중...")
        new_segments = _merge_with_original_duration(
            aligned["segments"], srt_segments
        )

        write_srt(new_segments, output_srt_path)
        log(f"저장 완료: {output_srt_path}")

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
