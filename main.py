"""
SRT 자동 정렬기 - GUI
ASR 엔진(FasterWhisper / Qwen3-ASR) 기반 자막 생성 / 기존 자막 재정렬(wav2vec2)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import multiprocessing
import os
import queue
import sys
import time

# pythonw.exe 실행 시 stdout/stderr가 None이라 huggingface/tqdm 등의 print가 죽음
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_config(data: dict) -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError:
        pass

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

from aligner import LANGUAGE_OPTIONS, MODEL_OPTIONS, build_output_path

# 라벨 → 엔진 id (aligner.create_engine에 전달)
ENGINE_DISPLAY = {
    "FasterWhisper": "fasterwhisper",
    "Qwen3-ASR": "qwen3",
    "Together API": "together",
    "ElevenLabs Scribe": "elevenlabs",
}
QWEN3_MODEL_OPTIONS = ["0.6B", "1.7B"]
ELEVENLABS_MODEL_OPTIONS = ["scribe_v2", "scribe_v1"]

# ── 색상/폰트 상수 ────────────────────────────────────────────────────────────
BG = "#1e1e2e"
BG2 = "#2a2a3e"
ACCENT = "#7c6af7"
ACCENT_HOVER = "#6a58e0"
FG = "#cdd6f4"
FG2 = "#a6adc8"
SUCCESS = "#a6e3a1"
ERROR = "#f38ba8"
FONT = ("Malgun Gothic", 10)
FONT_BOLD = ("Malgun Gothic", 10, "bold")
FONT_TITLE = ("Malgun Gothic", 13, "bold")
FONT_LOG = ("Malgun Gothic", 9)

MODE_GENERATE = "자막 생성"
MODE_ALIGN = "정렬만"


# ── 최상위 worker 함수 (multiprocessing pickle 요건) ─────────────────────────

def _worker_generate(log_queue, resp_queue, media, output_folder,
                     lang_code, model_size, max_chars, save_txt,
                     engine_id, qwen3_model, elevenlabs_model, elevenlabs_diarize):
    import warnings
    warnings.filterwarnings("ignore")
    from aligner import create_engine, transcribe_and_align

    def log(msg): log_queue.put(("normal", msg))
    def progress(v): log_queue.put(("__progress__", v))
    def confirm_overwrite(path):
        log_queue.put(("__ask_overwrite__", path))
        return resp_queue.get()

    try:
        engine = create_engine(
            engine_id,
            model_size=model_size,
            qwen3_model=qwen3_model,
            elevenlabs_model=elevenlabs_model,
            diarize=elevenlabs_diarize,
        )
        transcribe_and_align(
            media_path=media,
            output_folder=output_folder,
            language_code=lang_code,
            model_size=model_size,
            max_chars=max_chars,
            save_txt=save_txt,
            log=log,
            progress=progress,
            confirm_overwrite=confirm_overwrite,
            engine=engine,
        )
        log_queue.put(("success", "✓ 완료! 파일이 저장되었습니다."))
    except Exception as e:
        log_queue.put(("error", f"✗ 오류: {e}"))
    finally:
        log_queue.put(("__done__", ""))


def _worker_align(log_queue, resp_queue, media, srt, output_folder,
                  lang_code, max_chars, save_txt):
    import warnings
    warnings.filterwarnings("ignore")
    from aligner import align_srt

    def log(msg): log_queue.put(("normal", msg))
    def progress(v): log_queue.put(("__progress__", v))
    def confirm_overwrite(path):
        log_queue.put(("__ask_overwrite__", path))
        return resp_queue.get()

    try:
        align_srt(
            media_path=media,
            srt_path=srt,
            output_folder=output_folder,
            language_code=lang_code,
            max_chars=max_chars,
            save_txt=save_txt,
            log=log,
            progress=progress,
            confirm_overwrite=confirm_overwrite,
        )
        log_queue.put(("success", "✓ 완료! 파일이 저장되었습니다."))
    except Exception as e:
        log_queue.put(("error", f"✗ 오류: {e}"))
    finally:
        log_queue.put(("__done__", ""))


# ── App ───────────────────────────────────────────────────────────────────────

class App(TkinterDnD.Tk if _DND_AVAILABLE else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SRT 자동 정렬기")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._config = _load_config()

        # 이전 세션 상태 복원 — 잘못된 값/허용 목록 밖이면 기본값으로 fallback.
        # media_path / srt_path는 의도적으로 복원하지 않음 (영상마다 새로 선택)
        cfg = self._config

        def _pick_str(key, default, allowed):
            v = cfg.get(key)
            return v if isinstance(v, str) and v in allowed else default

        def _pick_bool(key, default):
            v = cfg.get(key)
            return v if isinstance(v, bool) else default

        def _pick_int(key, default, min_v, max_v):
            v = cfg.get(key)
            return v if isinstance(v, int) and min_v <= v <= max_v else default

        self._mode = tk.StringVar(value=_pick_str("mode", MODE_GENERATE, {MODE_GENERATE, MODE_ALIGN}))
        self._media_path = tk.StringVar()
        self._srt_path = tk.StringVar()
        self._output_folder = tk.StringVar()
        self._language = tk.StringVar(value=_pick_str("language", "자동 감지", set(LANGUAGE_OPTIONS.keys())))
        self._engine = tk.StringVar(value=_pick_str("engine", "FasterWhisper", set(ENGINE_DISPLAY.keys())))
        self._model_size = tk.StringVar(value=_pick_str("model_size", "large-v3", set(MODEL_OPTIONS)))
        self._qwen3_model = tk.StringVar(value=_pick_str("qwen3_model", "0.6B", set(QWEN3_MODEL_OPTIONS)))
        self._together_api_key = tk.StringVar(value=cfg.get("together_api_key", ""))
        self._elevenlabs_model = tk.StringVar(value=_pick_str("elevenlabs_model", "scribe_v2", set(ELEVENLABS_MODEL_OPTIONS)))
        self._elevenlabs_api_key = tk.StringVar(value=cfg.get("elevenlabs_api_key", ""))
        self._elevenlabs_diarize = tk.BooleanVar(value=_pick_bool("elevenlabs_diarize", False))
        self._split_enabled = tk.BooleanVar(value=_pick_bool("split_enabled", True))
        self._max_chars = tk.IntVar(value=_pick_int("max_chars", 84, 20, 120))
        self._save_txt = tk.BooleanVar(value=_pick_bool("save_txt", False))

        self._log_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._resp_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: multiprocessing.Process = None

        self._running = False
        self._start_time: float = 0.0
        self._timer_after_id = None

        self._build_ui()
        self._apply_engine_view()  # 복원된 engine에 맞춰 모델/키 위젯 동기화
        self._on_mode_change()
        self._poll_log()

        # 창 닫기 시 설정 저장
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 타이틀
        tk.Label(self, text="SRT 자동 정렬기", font=FONT_TITLE, bg=BG, fg=ACCENT
                 ).grid(row=0, column=0, columnspan=3, pady=(18, 4))
        tk.Label(self, text="faster-whisper 기반 자막 생성 / wav2vec2 싱크 정렬",
                 font=("Malgun Gothic", 9), bg=BG, fg=FG2
                 ).grid(row=1, column=0, columnspan=3, pady=(0, 10))

        # 모드 선택
        mode_frame = tk.Frame(self, bg=BG)
        mode_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        for mode in [MODE_GENERATE, MODE_ALIGN]:
            tk.Radiobutton(
                mode_frame,
                text=mode,
                variable=self._mode,
                value=mode,
                font=FONT_BOLD,
                bg=BG,
                fg=FG,
                selectcolor=BG2,
                activebackground=BG,
                activeforeground=ACCENT,
                command=self._on_mode_change,
            ).pack(side="left", padx=16)

        # 파일 입력
        self._file_row("영상 / 오디오", self._media_path, self._browse_media, 3,
                       accept_drop=[".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".flac", ".m4a", ".aac"])
        self._srt_widgets = self._file_row("입력 SRT", self._srt_path, self._browse_srt, 4,
                                           return_widgets=True, accept_drop=[".srt"])
        self._file_row("출력 폴더", self._output_folder, self._browse_output, 5)

        # 엔진 + 모델 선택 (생성 모드에서만 표시)
        self._model_label = tk.Label(self, text="엔진 / 모델", font=FONT_BOLD, bg=BG, fg=FG2, anchor="w", width=14)
        self._model_label.grid(row=6, column=0, sticky="w", padx=18, pady=5)

        self._engine_frame = tk.Frame(self, bg=BG)
        self._engine_frame.grid(row=6, column=1, columnspan=2, sticky="w", padx=(0, 18), pady=5)

        self._engine_cb = ttk.Combobox(
            self._engine_frame, textvariable=self._engine,
            values=list(ENGINE_DISPLAY.keys()), state="readonly", font=FONT, width=12,
        )
        self._engine_cb.pack(side="left", padx=(0, 8))
        self._engine_cb.bind("<<ComboboxSelected>>", lambda e: self._on_engine_change())

        self._model_cb = ttk.Combobox(
            self._engine_frame, textvariable=self._model_size,
            values=MODEL_OPTIONS, state="readonly", font=FONT, width=12,
        )
        self._qwen3_model_cb = ttk.Combobox(
            self._engine_frame, textvariable=self._qwen3_model,
            values=QWEN3_MODEL_OPTIONS, state="readonly", font=FONT, width=12,
        )
        # Together API: 모델 고정 + 키 설정 버튼 (모달에서 입력 → config.json에 저장)
        self._together_frame = tk.Frame(self._engine_frame, bg=BG)
        tk.Label(
            self._together_frame, text="whisper-large-v3",
            font=FONT, bg=BG, fg=FG2,
        ).pack(side="left", padx=(0, 8))
        self._together_key_btn = tk.Button(
            self._together_frame, text="🔑 키 설정",
            font=FONT, bg=BG2, fg=FG,
            activebackground=ACCENT, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=8,
            command=self._open_together_key_dialog,
        )
        self._together_key_btn.pack(side="left", padx=(0, 6))
        self._together_status_label = tk.Label(
            self._together_frame, text="",
            font=FONT, bg=BG, fg=FG2,
        )
        self._together_status_label.pack(side="left")
        self._refresh_together_status()

        # ElevenLabs: 모델 dropdown + 화자 분리 토글 + 키 설정 버튼
        self._elevenlabs_frame = tk.Frame(self._engine_frame, bg=BG)
        self._elevenlabs_model_cb = ttk.Combobox(
            self._elevenlabs_frame, textvariable=self._elevenlabs_model,
            values=ELEVENLABS_MODEL_OPTIONS, state="readonly", font=FONT, width=10,
        )
        self._elevenlabs_model_cb.pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            self._elevenlabs_frame, text="화자 분리",
            variable=self._elevenlabs_diarize,
            font=FONT, bg=BG, fg=FG,
            selectcolor=BG2, activebackground=BG, activeforeground=ACCENT,
        ).pack(side="left", padx=(0, 8))
        self._elevenlabs_key_btn = tk.Button(
            self._elevenlabs_frame, text="🔑 키 설정",
            font=FONT, bg=BG2, fg=FG,
            activebackground=ACCENT, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=8,
            command=self._open_elevenlabs_key_dialog,
        )
        self._elevenlabs_key_btn.pack(side="left", padx=(0, 6))
        self._elevenlabs_status_label = tk.Label(
            self._elevenlabs_frame, text="",
            font=FONT, bg=BG, fg=FG2,
        )
        self._elevenlabs_status_label.pack(side="left")
        self._refresh_elevenlabs_status()

        # 초기 엔진에 맞는 위젯은 _apply_engine_view가 __init__에서 호출되어 처리
        self._style_combobox()


        # 언어 선택
        tk.Label(self, text="언어", font=FONT_BOLD, bg=BG, fg=FG2, anchor="w", width=14
                 ).grid(row=7, column=0, sticky="w", padx=18, pady=5)
        lang_cb = ttk.Combobox(
            self, textvariable=self._language,
            values=list(LANGUAGE_OPTIONS.keys()), state="readonly", font=FONT, width=16,
        )
        lang_cb.grid(row=7, column=1, sticky="w", padx=(0, 6), pady=5)

        # 긴 자막 분할
        split_frame = tk.Frame(self, bg=BG)
        split_frame.grid(row=8, column=0, columnspan=3, padx=18, pady=(0, 4), sticky="w")

        tk.Checkbutton(
            split_frame, text="긴 자막 자동 분할",
            variable=self._split_enabled,
            font=FONT_BOLD, bg=BG, fg=FG,
            selectcolor=BG2, activebackground=BG, activeforeground=ACCENT,
            command=self._on_split_toggle,
        ).pack(side="left")

        tk.Label(split_frame, text="최대", font=FONT, bg=BG, fg=FG2).pack(side="left", padx=(16, 4))
        self._chars_spin = tk.Spinbox(
            split_frame, textvariable=self._max_chars,
            from_=20, to=120, increment=1, width=4,
            font=FONT, bg=BG2, fg=FG, buttonbackground=BG2,
            relief="flat", highlightthickness=0,
        )
        self._chars_spin.pack(side="left")
        tk.Label(split_frame, text="자", font=FONT, bg=BG, fg=FG2).pack(side="left", padx=(4, 0))

        tk.Checkbutton(
            split_frame, text="TXT도 저장",
            variable=self._save_txt,
            font=FONT_BOLD, bg=BG, fg=FG,
            selectcolor=BG2, activebackground=BG, activeforeground=ACCENT,
        ).pack(side="left", padx=(20, 0))

        # 시작 / 취소 버튼
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=(10, 6))

        self._run_btn = tk.Button(
            btn_frame, text="시작", font=FONT_BOLD,
            bg=ACCENT, fg="#ffffff",
            activebackground=ACCENT_HOVER, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=28, pady=8,
            command=self._start,
        )
        self._run_btn.pack(side="left", padx=8)

        self._cancel_btn = tk.Button(
            btn_frame, text="취소", font=FONT_BOLD,
            bg=BG2, fg=FG2,
            activebackground=ERROR, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=28, pady=8,
            state="disabled",
            command=self._cancel,
        )
        self._cancel_btn.pack(side="left", padx=8)

        # 진행 바 + 경과 시간
        bottom_frame = tk.Frame(self, bg=BG)
        bottom_frame.grid(row=10, column=0, columnspan=3, padx=18, pady=(0, 6), sticky="ew")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("custom.Horizontal.TProgressbar",
                        troughcolor=BG2, background=ACCENT,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
        self._progress = ttk.Progressbar(
            bottom_frame, style="custom.Horizontal.TProgressbar",
            mode="determinate", length=300, maximum=100,
        )
        self._progress.pack(side="left")

        self._percent_label = tk.Label(
            bottom_frame, text="", font=FONT, bg=BG, fg=ACCENT, width=5, anchor="w"
        )
        self._percent_label.pack(side="left", padx=(6, 0))

        self._timer_label = tk.Label(
            bottom_frame, text="", font=FONT, bg=BG, fg=FG2, width=8, anchor="e"
        )
        self._timer_label.pack(side="right")

        # 로그 창
        log_frame = tk.Frame(self, bg=BG2)
        log_frame.grid(row=11, column=0, columnspan=3, padx=18, pady=(0, 18), sticky="nsew")
        self._log_text = tk.Text(
            log_frame, height=12, width=62, font=FONT_LOG,
            bg=BG2, fg=FG, insertbackground=FG,
            relief="flat", state="disabled", wrap="word",
        )
        sb = tk.Scrollbar(log_frame, command=self._log_text.yview, bg=BG2)
        self._log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self._log_text.tag_configure("success", foreground=SUCCESS)
        self._log_text.tag_configure("error", foreground=ERROR)
        self._log_text.tag_configure("normal", foreground=FG)

    def _file_row(self, label, var, browse_cmd, row, return_widgets=False, accept_drop=None):
        lbl = tk.Label(self, text=label, font=FONT_BOLD, bg=BG, fg=FG2, anchor="w", width=14)
        lbl.grid(row=row, column=0, sticky="w", padx=18, pady=5)
        entry = tk.Entry(self, textvariable=var, font=FONT, bg=BG2, fg=FG,
                         insertbackground=FG, relief="flat", width=36)
        entry.grid(row=row, column=1, padx=(0, 6), pady=5)
        if _DND_AVAILABLE and accept_drop is not None:
            entry.drop_target_register(DND_FILES)
            entry.dnd_bind("<<Drop>>", lambda e: self._on_drop(e, var, accept_drop))
        btn = tk.Button(self, text="찾아보기", font=FONT, bg=BG2, fg=FG,
                        activebackground=ACCENT, activeforeground="#ffffff",
                        relief="flat", cursor="hand2", padx=8, command=browse_cmd)
        btn.grid(row=row, column=2, padx=(0, 18), pady=5)
        if return_widgets:
            return lbl, entry, btn

    def _on_drop(self, event, var, accept_exts):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            path = raw[1:-1]
        else:
            path = raw.split()[0]
        ext = os.path.splitext(path)[1].lower()
        if accept_exts and ext not in accept_exts:
            messagebox.showwarning("파일 형식 오류", f"지원하지 않는 파일 형식입니다: {ext}")
            return
        var.set(path)
        if var in (self._media_path, self._srt_path) and not self._output_folder.get():
            self._output_folder.set(os.path.dirname(path))

    def _style_combobox(self):
        style = ttk.Style()
        style.configure("TCombobox", fieldbackground=BG2, background=BG2,
                        foreground=FG, selectbackground=ACCENT,
                        selectforeground="#ffffff", arrowcolor=FG2)

    # ── 모드 전환 ─────────────────────────────────────────────────────────────

    def _on_split_toggle(self):
        state = "normal" if self._split_enabled.get() else "disabled"
        self._chars_spin.config(state=state)

    def _on_mode_change(self):
        mode = self._mode.get()
        is_generate = mode == MODE_GENERATE
        is_align_only = mode == MODE_ALIGN

        for widget in self._srt_widgets:
            if is_align_only:
                widget.grid()
            else:
                widget.grid_remove()

        if is_generate:
            self._model_label.grid()
            self._engine_frame.grid()
        else:
            self._model_label.grid_remove()
            self._engine_frame.grid_remove()

    def _apply_engine_view(self):
        """현재 _engine 값에 맞춰 모델 dropdown / API 키 위젯만 교체 (max_chars 불변).

        초기 로드 시 호출하면 저장된 max_chars를 덮어쓰지 않고 UI만 동기화 가능.
        """
        eng = self._engine_id()
        self._model_cb.pack_forget()
        self._qwen3_model_cb.pack_forget()
        self._together_frame.pack_forget()
        self._elevenlabs_frame.pack_forget()

        if eng == "qwen3":
            self._qwen3_model_cb.pack(side="left")
        elif eng == "together":
            self._together_frame.pack(side="left")
        elif eng == "elevenlabs":
            self._elevenlabs_frame.pack(side="left")
        else:  # fasterwhisper
            self._model_cb.pack(side="left")

    def _on_engine_change(self):
        """엔진 콤보박스를 사용자가 직접 바꿨을 때: UI 교체 + 권장 max_chars 자동 적용.

        엔진별 권장값 (사용자가 후속 수정 가능):
          - ElevenLabs Scribe: 42자 (BBC/Netflix 표준) — Scribe sentence segment가
            한 화면 자막으로 너무 길어지는 것 방지
          - 그 외: 84자 (기존 기본)
        """
        self._apply_engine_view()
        self._max_chars.set(42 if self._engine_id() == "elevenlabs" else 84)

    def _save_together_key(self):
        self._config["together_api_key"] = self._together_api_key.get().strip()
        _save_config(self._config)
        self._refresh_together_status()

    def _refresh_together_status(self):
        has_key = bool(self._together_api_key.get().strip())
        text = "✓ 키 설정됨" if has_key else "✗ 키 미설정"
        color = SUCCESS if has_key else ERROR
        self._together_status_label.config(text=text, fg=color)

    def _open_together_key_dialog(self):
        """Together API 키 입력 모달 — 입력 → 저장 후 닫힘."""
        dlg = tk.Toplevel(self)
        dlg.title("Together API 키 설정")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        # 부모 창 기준 중앙 배치
        self.update_idletasks()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw, ph = self.winfo_width(), self.winfo_height()
        dw, dh = 420, 200
        dlg.geometry(f"{dw}x{dh}+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

        tk.Label(
            dlg, text="Together API 키",
            font=FONT_BOLD, bg=BG, fg=FG,
        ).pack(pady=(18, 4))
        tk.Label(
            dlg, text="https://api.together.ai/settings/api-keys 에서 발급",
            font=("Malgun Gothic", 9), bg=BG, fg=FG2,
        ).pack(pady=(0, 12))

        local_var = tk.StringVar(value=self._together_api_key.get())
        show_var = tk.BooleanVar(value=False)

        entry_frame = tk.Frame(dlg, bg=BG)
        entry_frame.pack(pady=(0, 8))

        entry = tk.Entry(
            entry_frame, textvariable=local_var,
            show="*", font=FONT, bg=BG2, fg=FG, insertbackground=FG,
            relief="flat", width=38,
        )
        entry.pack(side="left", padx=(0, 6))
        entry.focus_set()

        def toggle_show():
            entry.config(show="" if show_var.get() else "*")

        tk.Checkbutton(
            entry_frame, text="보이기",
            variable=show_var, command=toggle_show,
            font=("Malgun Gothic", 9), bg=BG, fg=FG2,
            selectcolor=BG2, activebackground=BG, activeforeground=ACCENT,
        ).pack(side="left")

        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(pady=(14, 18))

        def on_save():
            self._together_api_key.set(local_var.get().strip())
            self._save_together_key()
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        tk.Button(
            btn_frame, text="저장", font=FONT_BOLD,
            bg=ACCENT, fg="#ffffff",
            activebackground=ACCENT_HOVER, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=20, pady=6,
            command=on_save,
        ).pack(side="left", padx=6)
        tk.Button(
            btn_frame, text="취소", font=FONT,
            bg=BG2, fg=FG2,
            activebackground=ERROR, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=20, pady=6,
            command=on_cancel,
        ).pack(side="left", padx=6)

        dlg.bind("<Return>", lambda e: on_save())
        dlg.bind("<Escape>", lambda e: on_cancel())

    def _save_elevenlabs_key(self):
        self._config["elevenlabs_api_key"] = self._elevenlabs_api_key.get().strip()
        _save_config(self._config)
        self._refresh_elevenlabs_status()

    def _refresh_elevenlabs_status(self):
        has_key = bool(self._elevenlabs_api_key.get().strip())
        text = "✓ 키 설정됨" if has_key else "✗ 키 미설정"
        color = SUCCESS if has_key else ERROR
        self._elevenlabs_status_label.config(text=text, fg=color)

    def _open_elevenlabs_key_dialog(self):
        """ElevenLabs API 키 입력 모달 — 입력 → 저장 후 닫힘."""
        dlg = tk.Toplevel(self)
        dlg.title("ElevenLabs API 키 설정")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        # 부모 창 기준 중앙 배치
        self.update_idletasks()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw, ph = self.winfo_width(), self.winfo_height()
        dw, dh = 420, 200
        dlg.geometry(f"{dw}x{dh}+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

        tk.Label(
            dlg, text="ElevenLabs API 키",
            font=FONT_BOLD, bg=BG, fg=FG,
        ).pack(pady=(18, 4))
        tk.Label(
            dlg, text="https://elevenlabs.io/app/settings/api-keys 에서 발급",
            font=("Malgun Gothic", 9), bg=BG, fg=FG2,
        ).pack(pady=(0, 12))

        local_var = tk.StringVar(value=self._elevenlabs_api_key.get())
        show_var = tk.BooleanVar(value=False)

        entry_frame = tk.Frame(dlg, bg=BG)
        entry_frame.pack(pady=(0, 8))

        entry = tk.Entry(
            entry_frame, textvariable=local_var,
            show="*", font=FONT, bg=BG2, fg=FG, insertbackground=FG,
            relief="flat", width=38,
        )
        entry.pack(side="left", padx=(0, 6))
        entry.focus_set()

        def toggle_show():
            entry.config(show="" if show_var.get() else "*")

        tk.Checkbutton(
            entry_frame, text="보이기",
            variable=show_var, command=toggle_show,
            font=("Malgun Gothic", 9), bg=BG, fg=FG2,
            selectcolor=BG2, activebackground=BG, activeforeground=ACCENT,
        ).pack(side="left")

        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(pady=(14, 18))

        def on_save():
            self._elevenlabs_api_key.set(local_var.get().strip())
            self._save_elevenlabs_key()
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        tk.Button(
            btn_frame, text="저장", font=FONT_BOLD,
            bg=ACCENT, fg="#ffffff",
            activebackground=ACCENT_HOVER, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=20, pady=6,
            command=on_save,
        ).pack(side="left", padx=6)
        tk.Button(
            btn_frame, text="취소", font=FONT,
            bg=BG2, fg=FG2,
            activebackground=ERROR, activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=20, pady=6,
            command=on_cancel,
        ).pack(side="left", padx=6)

        dlg.bind("<Return>", lambda e: on_save())
        dlg.bind("<Escape>", lambda e: on_cancel())

    def _save_ui_settings(self):
        """현재 UI 상태를 config.json에 저장 (API 키 등 다른 키는 보존).

        창 종료 시점과 _start 진입 시점에 호출 — 작업 도중 크래시가 나도 직전 설정은
        남아 있게 함.
        """
        self._config["mode"] = self._mode.get()
        self._config["engine"] = self._engine.get()
        self._config["model_size"] = self._model_size.get()
        self._config["qwen3_model"] = self._qwen3_model.get()
        self._config["elevenlabs_model"] = self._elevenlabs_model.get()
        self._config["elevenlabs_diarize"] = self._elevenlabs_diarize.get()
        self._config["language"] = self._language.get()
        self._config["split_enabled"] = self._split_enabled.get()
        self._config["max_chars"] = self._max_chars.get()
        self._config["save_txt"] = self._save_txt.get()
        self._config.pop("output_folder", None)
        _save_config(self._config)

    def _on_close(self):
        self._save_ui_settings()
        self.destroy()

    def _engine_id(self) -> str:
        return ENGINE_DISPLAY.get(self._engine.get(), "fasterwhisper")

    # ── 파일 다이얼로그 ───────────────────────────────────────────────────────

    def _browse_media(self):
        path = filedialog.askopenfilename(
            title="영상 또는 오디오 파일 선택",
            filetypes=[
                ("미디어 파일", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.m4a *.aac"),
                ("모든 파일", "*.*"),
            ],
        )
        if path:
            self._media_path.set(path)
            if not self._output_folder.get():
                self._output_folder.set(os.path.dirname(path))

    def _browse_srt(self):
        path = filedialog.askopenfilename(
            title="입력 SRT 파일 선택",
            filetypes=[("SRT 자막", "*.srt"), ("모든 파일", "*.*")],
        )
        if path:
            self._srt_path.set(path)
            if not self._output_folder.get():
                self._output_folder.set(os.path.dirname(path))

    def _browse_output(self):
        folder = filedialog.askdirectory(title="출력 폴더 선택")
        if folder:
            self._output_folder.set(folder)

    # ── 실행 로직 ─────────────────────────────────────────────────────────────

    def _validate(self) -> bool:
        if not self._media_path.get() or not os.path.isfile(self._media_path.get()):
            messagebox.showerror("오류", "영상/오디오 파일을 선택하세요.")
            return False
        if self._mode.get() == MODE_ALIGN:
            if not self._srt_path.get() or not os.path.isfile(self._srt_path.get()):
                messagebox.showerror("오류", "입력 SRT 파일을 선택하세요.")
                return False

        if not self._output_folder.get() or not os.path.isdir(self._output_folder.get()):
            messagebox.showerror("오류", "출력 폴더를 지정하세요.")
            return False

        # Together API: 시작 전 키 확인 (입력란 → 환경변수 순)
        if self._mode.get() == MODE_GENERATE and self._engine_id() == "together":
            key = self._together_api_key.get().strip() or os.environ.get("TOGETHER_API_KEY", "").strip()
            if not key:
                messagebox.showerror(
                    "오류",
                    "Together API 키가 없습니다.\n'🔑 키 설정' 버튼으로 입력하세요.",
                )
                return False
            # worker(spawn) 프로세스가 상속받도록 부모 환경에 주입
            os.environ["TOGETHER_API_KEY"] = key

        # ElevenLabs Scribe: 시작 전 키 확인 (입력란 → 환경변수 순)
        if self._mode.get() == MODE_GENERATE and self._engine_id() == "elevenlabs":
            key = self._elevenlabs_api_key.get().strip() or os.environ.get("ELEVENLABS_API_KEY", "").strip()
            if not key:
                messagebox.showerror(
                    "오류",
                    "ElevenLabs API 키가 없습니다.\n'🔑 키 설정' 버튼으로 입력하세요.",
                )
                return False
            os.environ["ELEVENLABS_API_KEY"] = key

        return True

    def _start(self):
        if self._running or not self._validate():
            return

        # 작업 시작 직전 상태 저장 — 도중에 크래시가 나도 직전 설정은 다음 실행에 반영
        self._save_ui_settings()

        lang_code = LANGUAGE_OPTIONS.get(self._language.get())

        if lang_code is not None:
            preview_path = build_output_path(
                self._output_folder.get(), self._media_path.get(), lang_code
            )
            if os.path.exists(preview_path):
                if not messagebox.askyesno(
                    "파일 덮어쓰기",
                    f"이미 존재하는 파일입니다:\n{preview_path}\n\n덮어쓰겠습니까?",
                ):
                    return

        # 매 실행마다 큐 새로 생성 (이전 잔여 메시지 방지)
        self._log_queue = multiprocessing.Queue()
        self._resp_queue = multiprocessing.Queue()

        self._running = True
        self._run_btn.config(state="disabled", text="처리 중...")
        self._cancel_btn.config(state="normal", bg=ERROR, fg="#ffffff")
        self._progress["value"] = 0
        self._percent_label.config(text="")
        self._start_time = time.time()
        self._tick_timer()
        self._clear_log()

        max_chars = self._max_chars.get() if self._split_enabled.get() else 0

        if self._mode.get() == MODE_GENERATE:
            self._process = multiprocessing.Process(
                target=_worker_generate,
                args=(self._log_queue, self._resp_queue,
                      self._media_path.get(), self._output_folder.get(),
                      lang_code, self._model_size.get(),
                      max_chars, self._save_txt.get(),
                      self._engine_id(), self._qwen3_model.get(),
                      self._elevenlabs_model.get(), self._elevenlabs_diarize.get()),
                daemon=True,
            )
        else:
            # mode 2(정렬만)는 Qwen3 미지원 — 항상 FasterWhisper 사용
            self._process = multiprocessing.Process(
                target=_worker_align,
                args=(self._log_queue, self._resp_queue,
                      self._media_path.get(), self._srt_path.get(),
                      self._output_folder.get(), lang_code, max_chars, self._save_txt.get()),
                daemon=True,
            )

        self._process.start()

    def _cancel(self):
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self._log_queue.put(("error", "✗ 취소되었습니다."))
        self._log_queue.put(("__cancelled__", ""))

    # ── 타이머 ────────────────────────────────────────────────────────────────

    def _tick_timer(self):
        if not self._running:
            return
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        self._timer_label.config(text=f"{m:02d}:{s:02d}")
        self._timer_after_id = self.after(1000, self._tick_timer)

    # ── 로그 폴링 ─────────────────────────────────────────────────────────────

    def _finalize(self, completed: bool = True):
        if self._timer_after_id:
            self.after_cancel(self._timer_after_id)
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        self._timer_label.config(text=f"{m:02d}:{s:02d}")
        if completed:
            self._progress["value"] = 100
            self._percent_label.config(text="100%")
        self._run_btn.config(state="normal", text="시작")
        self._cancel_btn.config(state="disabled", bg=BG2, fg=FG2)
        self._running = False

    def _poll_log(self):
        try:
            while True:
                tag, msg = self._log_queue.get_nowait()
                if tag == "__done__":
                    self._finalize(completed=True)
                elif tag == "__cancelled__":
                    self._finalize(completed=False)
                elif tag == "__progress__":
                    self._progress["value"] = msg
                    self._percent_label.config(text=f"{int(msg)}%")
                elif tag == "__ask_overwrite__":
                    answer = messagebox.askyesno(
                        "파일 덮어쓰기",
                        f"이미 존재하는 파일입니다:\n{msg}\n\n덮어쓰겠습니까?",
                    )
                    self._resp_queue.put(answer)
                else:
                    self._append_log(msg, tag)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _append_log(self, msg: str, tag: str = "normal"):
        self._log_text.config(state="normal")
        self._log_text.insert("end", msg + "\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
