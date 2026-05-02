"""
Qwen3-ASR 헤드리스 러너 — Qwen3Engine이 별도 venv subprocess로 실행.

부모 프로세스(SRT-auto-aligner)는 qwen-asr이 설치된 venv의 python.exe로 본 스크립트를 호출한다.
stdout: 단일 JSON 객체 ({ok, language_detected(ISO), text, items[{text, start, end}]})
stderr: [STAGE] / [PROGRESS N] 라인으로 진행 보고
exit code: 0 성공 / 1 실패 (실패 시에도 stdout에 {ok: false, error: ...} 1줄 기록)
"""
import argparse
import json
import os
import sys
import time
import traceback


# ISO 코드 ↔ Qwen3 언어 이름 (Qwen3는 full name, WhisperX는 ISO)
_ISO_TO_QWEN_NAME = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian",
    "pt": "Portuguese", "it": "Italian", "ar": "Arabic",
}
_QWEN_NAME_TO_ISO = {v: k for k, v in _ISO_TO_QWEN_NAME.items()}


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", file=sys.stderr, flush=True)


def _progress(pct: int) -> None:
    print(f"[PROGRESS] {pct}", file=sys.stderr, flush=True)


def _emit_result(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-ASR headless runner")
    ap.add_argument("--audio", required=True, help="오디오 파일 (wav 16kHz mono 권장)")
    ap.add_argument("--language", default="", help="ISO 코드 (en/ko/...) 또는 빈 문자열로 자동 감지")
    ap.add_argument("--model", default="0.6B", choices=["0.6B", "1.7B"])
    args = ap.parse_args()

    audio_path = args.audio
    if not os.path.isfile(audio_path):
        _emit_result({"ok": False, "error": f"오디오 파일 없음: {audio_path}"})
        return 1

    # nagisa C 확장(_dynet)이 Windows에서 한국어 경로를 못 읽으므로 ASCII 강제
    try:
        audio_path.encode("ascii")
    except UnicodeEncodeError:
        _emit_result({
            "ok": False,
            "error": (
                f"Qwen3는 ASCII 경로만 지원합니다 (받은 경로: {audio_path}). "
                "한글 경로 환경에서는 WhisperX 엔진을 사용하세요."
            ),
        })
        return 1

    qwen_lang = _ISO_TO_QWEN_NAME.get(args.language) if args.language else None

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except Exception as e:
        _emit_result({"ok": False, "error": f"qwen-asr 모듈 임포트 실패: {e}"})
        return 1

    asr_model_id = f"Qwen/Qwen3-ASR-{args.model}"
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    _stage(f"Qwen3 모델 로드 중 ({args.model})")
    _progress(35)
    t0 = time.time()
    try:
        model = Qwen3ASRModel.from_pretrained(
            asr_model_id,
            dtype=torch.bfloat16,
            device_map=device_map,
            max_inference_batch_size=8,
            max_new_tokens=512,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device_map,
            ),
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        _emit_result({"ok": False, "error": f"Qwen3 모델 로드 실패: {e}"})
        return 1
    _stage(f"모델 로드 완료 ({time.time() - t0:.1f}s)")
    _progress(55)

    _stage("전사 + 정렬 중...")
    _progress(60)
    t0 = time.time()
    try:
        results = model.transcribe(
            audio=str(audio_path),
            language=qwen_lang,
            return_time_stamps=True,
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        _emit_result({"ok": False, "error": f"Qwen3 전사 실패: {e}"})
        return 1
    _stage(f"전사 완료 ({time.time() - t0:.1f}s)")
    _progress(85)

    r = results[0]
    detected_full = getattr(r, "language", None)
    detected_iso = _QWEN_NAME_TO_ISO.get(detected_full or "", args.language or "en")

    timestamps = getattr(r, "time_stamps", None)
    items_raw = []
    if timestamps is not None:
        if hasattr(timestamps, "items") and not isinstance(timestamps, dict):
            items_raw = timestamps.items
        elif isinstance(timestamps, list):
            items_raw = timestamps
        elif isinstance(timestamps, dict):
            items_raw = timestamps.get("items") or next(
                (v for v in timestamps.values() if isinstance(v, list)), [])
        elif hasattr(timestamps, "__dict__"):
            d = vars(timestamps)
            items_raw = d.get("items") or next(
                (v for v in d.values() if isinstance(v, list)), [])

    items_clean = []
    for it in items_raw:
        if hasattr(it, "__dict__") and not isinstance(it, dict):
            it = vars(it)
        if not isinstance(it, dict):
            continue
        text = it.get("text") or it.get("word")
        st = it.get("start_time", it.get("start"))
        et = it.get("end_time", it.get("end"))
        if text is None or st is None or et is None:
            continue
        items_clean.append({"text": str(text), "start": float(st), "end": float(et)})

    _progress(90)
    _emit_result({
        "ok": True,
        "language_detected": detected_iso,
        "text": r.text,
        "items": items_clean,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
