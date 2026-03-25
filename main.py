"""
SRT 자동 정렬기 - GUI
WhisperX (wav2vec2) 기반으로 자막 타임스탬프를 오디오에 맞춰 재정렬합니다.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from aligner import LANGUAGE_OPTIONS, align_srt

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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SRT 자동 정렬기")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._media_path = tk.StringVar()
        self._srt_path = tk.StringVar()
        self._output_path = tk.StringVar()
        self._language = tk.StringVar(value="자동 감지")
        self._log_queue: queue.Queue = queue.Queue()
        self._running = False

        self._build_ui()
        self._poll_log()

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 18, "pady": 6}

        # 타이틀
        tk.Label(
            self,
            text="SRT 자동 정렬기",
            font=FONT_TITLE,
            bg=BG,
            fg=ACCENT,
        ).grid(row=0, column=0, columnspan=3, pady=(18, 4))

        tk.Label(
            self,
            text="Whisper 자막의 싱크를 wav2vec2로 오디오에 맞춰 재정렬합니다.",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG2,
        ).grid(row=1, column=0, columnspan=3, pady=(0, 14))

        # 파일 입력 영역
        self._file_row("영상 / 오디오", self._media_path, self._browse_media, 2)
        self._file_row("입력 SRT", self._srt_path, self._browse_srt, 3)
        self._file_row("출력 SRT", self._output_path, self._browse_output, 4)

        # 언어 선택
        tk.Label(self, text="언어", font=FONT_BOLD, bg=BG, fg=FG2, anchor="w", width=14).grid(
            row=5, column=0, sticky="w", **pad
        )
        lang_cb = ttk.Combobox(
            self,
            textvariable=self._language,
            values=list(LANGUAGE_OPTIONS.keys()),
            state="readonly",
            font=FONT,
            width=28,
        )
        lang_cb.grid(row=5, column=1, sticky="w", **pad)
        self._style_combobox(lang_cb)

        # 실행 버튼
        self._run_btn = tk.Button(
            self,
            text="정렬 시작",
            font=FONT_BOLD,
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            padx=24,
            pady=8,
            command=self._start,
        )
        self._run_btn.grid(row=6, column=0, columnspan=3, pady=(10, 6))

        # 진행 바
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "custom.Horizontal.TProgressbar",
            troughcolor=BG2,
            background=ACCENT,
            bordercolor=BG,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        self._progress = ttk.Progressbar(
            self, style="custom.Horizontal.TProgressbar", mode="indeterminate", length=400
        )
        self._progress.grid(row=7, column=0, columnspan=3, padx=18, pady=(0, 6))

        # 로그 창
        log_frame = tk.Frame(self, bg=BG2, bd=0)
        log_frame.grid(row=8, column=0, columnspan=3, padx=18, pady=(0, 18), sticky="nsew")

        self._log_text = tk.Text(
            log_frame,
            height=12,
            width=60,
            font=FONT_LOG,
            bg=BG2,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            state="disabled",
            wrap="word",
        )
        scrollbar = tk.Scrollbar(log_frame, command=self._log_text.yview, bg=BG2)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self._log_text.tag_configure("success", foreground=SUCCESS)
        self._log_text.tag_configure("error", foreground=ERROR)
        self._log_text.tag_configure("normal", foreground=FG)

    def _file_row(self, label: str, var: tk.StringVar, browse_cmd, row: int):
        tk.Label(self, text=label, font=FONT_BOLD, bg=BG, fg=FG2, anchor="w", width=14).grid(
            row=row, column=0, sticky="w", padx=18, pady=5
        )
        entry = tk.Entry(
            self,
            textvariable=var,
            font=FONT,
            bg=BG2,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            width=36,
        )
        entry.grid(row=row, column=1, padx=(0, 6), pady=5)
        tk.Button(
            self,
            text="찾아보기",
            font=FONT,
            bg=BG2,
            fg=FG,
            activebackground=ACCENT,
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            padx=8,
            command=browse_cmd,
        ).grid(row=row, column=2, padx=(0, 18), pady=5)

    @staticmethod
    def _style_combobox(cb: ttk.Combobox):
        style = ttk.Style()
        style.configure(
            "TCombobox",
            fieldbackground=BG2,
            background=BG2,
            foreground=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            arrowcolor=FG2,
        )

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
            if not self._output_path.get():
                base, _ = os.path.splitext(path)
                self._output_path.set(base + "_aligned.srt")

    def _browse_srt(self):
        path = filedialog.askopenfilename(
            title="입력 SRT 파일 선택",
            filetypes=[("SRT 자막", "*.srt"), ("모든 파일", "*.*")],
        )
        if path:
            self._srt_path.set(path)
            if not self._output_path.get():
                base, _ = os.path.splitext(path)
                self._output_path.set(base + "_aligned.srt")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="출력 SRT 저장 위치",
            defaultextension=".srt",
            filetypes=[("SRT 자막", "*.srt")],
        )
        if path:
            self._output_path.set(path)

    # ── 실행 로직 ─────────────────────────────────────────────────────────────

    def _validate(self) -> bool:
        if not self._media_path.get():
            messagebox.showerror("오류", "영상/오디오 파일을 선택하세요.")
            return False
        if not os.path.isfile(self._media_path.get()):
            messagebox.showerror("오류", "영상/오디오 파일을 찾을 수 없습니다.")
            return False
        if not self._srt_path.get():
            messagebox.showerror("오류", "입력 SRT 파일을 선택하세요.")
            return False
        if not os.path.isfile(self._srt_path.get()):
            messagebox.showerror("오류", "SRT 파일을 찾을 수 없습니다.")
            return False
        if not self._output_path.get():
            messagebox.showerror("오류", "출력 SRT 경로를 지정하세요.")
            return False
        return True

    def _start(self):
        if self._running:
            return
        if not self._validate():
            return

        lang_label = self._language.get()
        lang_code = LANGUAGE_OPTIONS.get(lang_label)

        self._running = True
        self._run_btn.config(state="disabled", text="처리 중...")
        self._progress.start(12)
        self._clear_log()

        thread = threading.Thread(
            target=self._run_align,
            args=(
                self._media_path.get(),
                self._srt_path.get(),
                self._output_path.get(),
                lang_code,
            ),
            daemon=True,
        )
        thread.start()

    def _run_align(self, media, srt, output, lang_code):
        try:
            align_srt(
                media_path=media,
                srt_path=srt,
                output_srt_path=output,
                language_code=lang_code,
                log=lambda msg: self._log_queue.put(("normal", msg)),
            )
            self._log_queue.put(("success", "✓ 완료! 파일이 저장되었습니다."))
        except Exception as e:
            self._log_queue.put(("error", f"✗ 오류: {e}"))
        finally:
            self._log_queue.put(("__done__", ""))

    # ── 로그 폴링 ─────────────────────────────────────────────────────────────

    def _poll_log(self):
        try:
            while True:
                tag, msg = self._log_queue.get_nowait()
                if tag == "__done__":
                    self._progress.stop()
                    self._run_btn.config(state="normal", text="정렬 시작")
                    self._running = False
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
    app = App()
    app.mainloop()
