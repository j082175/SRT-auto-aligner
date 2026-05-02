"""
헤드리스 CLI 진입점 — CEP/외부 프로세스에서 호출용.

사용:
    python cli.py --input <video> --output <folder> [--engine fasterwhisper] [--model large-v3] [--language auto]

stdout: 마지막 줄에 "RESULT: <SRT 절대경로>"
stderr: 진행 로그, "[PROGRESS] <0~100>" 형식의 진행률
exit code: 0 성공 / 1 실행 오류 / 2 입력 오류 / 3 결과 경로 누락
"""

import argparse
import os
import sys

from aligner import FasterWhisperEngine, MODEL_OPTIONS, transcribe_and_align

ENGINE_CHOICES = ["fasterwhisper"]


def build_engine(name: str, model_size: str):
    if name == "fasterwhisper":
        return FasterWhisperEngine(model_size=model_size)
    raise ValueError(f"지원되지 않는 엔진: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="자막 헤드리스 생성 (faster-whisper)")
    parser.add_argument("--input", required=True, help="입력 영상 파일")
    parser.add_argument("--output", required=True, help="출력 폴더")
    parser.add_argument(
        "--engine", default="fasterwhisper", choices=ENGINE_CHOICES,
        help="ASR 엔진 (기본: fasterwhisper)",
    )
    parser.add_argument(
        "--model", default="large-v3", choices=MODEL_OPTIONS,
        help="Whisper 모델 크기 (fasterwhisper 엔진 전용, 기본: large-v3)",
    )
    parser.add_argument(
        "--language", default="auto",
        help="언어 코드 (ko/en/ja/zh/es/fr/de/ru/pt/it/ar) 또는 auto",
    )
    parser.add_argument(
        "--max-chars", type=int, default=84,
        help="긴 자막 분할 임계값 (0 = 분할 안 함, 기본 84 = GUI 앱과 동일)",
    )
    parser.add_argument(
        "--save-txt", action="store_true",
        help="SRT와 함께 .txt(텍스트만) 파일도 저장",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] 입력 파일이 없습니다: {args.input}", file=sys.stderr, flush=True)
        return 2

    os.makedirs(args.output, exist_ok=True)
    lang_code = None if args.language == "auto" else args.language

    captured = {"path": None}

    def log(msg: str) -> None:
        # [STAGE] prefix로 CEP가 의도된 메시지만 식별 (외부 라이브러리 warning과 분리)
        print(f"[STAGE] {msg}", file=sys.stderr, flush=True)
        if msg.startswith("저장 완료: "):
            # "저장 완료: <srt>, <txt>" 형식이면 첫 srt만 잡기
            tail = msg[len("저장 완료: "):].strip()
            captured["path"] = tail.split(",")[0].strip()

    def progress(pct: int) -> None:
        print(f"[PROGRESS] {pct}", file=sys.stderr, flush=True)

    try:
        engine = build_engine(args.engine, args.model)
        transcribe_and_align(
            media_path=args.input,
            output_folder=args.output,
            language_code=lang_code,
            model_size=args.model,
            max_chars=args.max_chars,
            save_txt=args.save_txt,
            log=log,
            progress=progress,
            confirm_overwrite=lambda _: True,
            engine=engine,
        )
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1

    if not captured["path"]:
        print("[ERROR] 결과 SRT 경로를 추출하지 못했습니다.", file=sys.stderr, flush=True)
        return 3

    print(f"RESULT: {captured['path']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
