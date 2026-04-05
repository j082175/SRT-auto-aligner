"""
SRT 자동 정렬기 - GUI
faster-whisper 기반 자막 생성 / 기존 자막 재정렬(wav2vec2)
"""

import warnings
warnings.filterwarnings("ignore")

import multiprocessing
import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

from aligner import LANGUAGE_OPTIONS, MODEL_OPTIONS, build_output_path

# ── 색상/폰트 상수 ────────────────────────────────────────────────────────────
BG = "#1e1e2e"
BG2 = "#2a2a3e"
ACCENT = "#7c6af7"
ACCENT_HOVER = "#6a58e0"
FG = "#cdd6f4"
FG2 = "#a6adc8"
SUCCESS = "#a6e3a1"
ERROR = "#f38ba8"
FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_LOG = ("Consolas", 9)

MODE_GENERATE = "자막 생성"
MODE_ALIGN = "정렬만"


# ── 최상위 worker 함수 (multiprocessing pickle 요건) ─────────────────────────

def _worker_generate(log_queue, resp_queue, media, output_folder,
                     lang_code, model_size, max_chars, save_txt):
    import warnings
    warnings.filterwarnings("ignore")
    from aligner import transcribe_and_align

    def log(msg): log_queue.put(("normal", msg))
    def progress(v): log_queue.put(("__progress__", v))
    def confirm_overwrite(path):
        log_queue.put(("__ask_overwrite__", path))
        return resp_queue.get()

    try:
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

        self._mode = tk.StringVar(value=MODE_GENERATE)
        self._media_path = tk.StringVar()
        self._srt_path = tk.StringVar()
        self._output_folder = tk.StringVar()
        self._language = tk.StringVar(value="자동 감지")
        self._model_size = tk.StringVar(value="large-v3")
        self._split_enabled = tk.BooleanVar(value=True)
        self._max_chars = tk.IntVar(value=84)
        self._save_txt = tk.BooleanVar(value=False)

        self._log_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._resp_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: multiprocessing.Process = None

        self._running = False
        self._start_time: float = 0.0
        self._timer_after_id = None

        self._build_ui()
        self._on_mode_change()
        self._poll_log()

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 타이틀
        tk.Label(self, text="SRT 자동 정렬기", font=FONT_TITLE, bg=BG, fg=ACCENT
                 ).grid(row=0, column=0, columnspan=3, pady=(18, 4))
        tk.Label(self, text="faster-whisper 기반 자막 생성 / wav2vec2 싱크 정렬",
                 font=("Segoe UI", 9), bg=BG, fg=FG2
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

        # 모델 선택 (생성 모드에서만 표시)
        self._model_label = tk.Label(self, text="Whisper 모델", font=FONT_BOLD, bg=BG, fg=FG2, anchor="w", width=14)
        self._model_label.grid(row=6, column=0, sticky="w", padx=18, pady=5)

        model_batch_frame = tk.Frame(self, bg=BG)
        model_batch_frame.grid(row=6, column=1, columnspan=2, sticky="w", padx=(0, 18), pady=5)

        self._model_cb = ttk.Combobox(
            model_batch_frame, textvariable=self._model_size,
            values=MODEL_OPTIONS, state="readonly", font=FONT, width=14,
        )
        self._model_cb.pack(side="left")
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
            self._model_cb.master.grid()
        else:
            self._model_label.grid_remove()
            self._model_cb.master.grid_remove()

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
        return True

    def _start(self):
        if self._running or not self._validate():
            return

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
                      max_chars, self._save_txt.get()),
                daemon=True,
            )
        else:
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
