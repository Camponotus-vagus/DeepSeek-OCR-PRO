"""GUI application using CustomTkinter."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from ..config import OCRConfig
from ..utils.logging_setup import setup_logging
from ..utils.paths import resolve_model_path


def launch_gui():
    """Launch the GUI application."""
    try:
        import customtkinter as ctk
    except ImportError:
        print("GUI requires customtkinter. Install with: pip install customtkinter")
        sys.exit(1)

    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    OCRApp(root)
    root.mainloop()


class OCRApp:
    """Modern OCR GUI application."""

    def __init__(self, root):
        import customtkinter as ctk

        self.root = root
        self.root.title("DeepSeek OCR PRO V2")
        self.root.geometry("1100x850")

        self._engine = None
        self._is_processing = False
        self._cancel_event = threading.Event()

        setup_logging(verbose=False)
        self._build_ui(ctk)

    def _build_ui(self, ctk):
        self.root.grid_columnconfigure(0, weight=1)

        # Header
        top = ctk.CTkFrame(self.root, fg_color="transparent")
        top.grid(row=0, column=0, pady=10, sticky="ew")

        ctk.CTkLabel(top, text="DeepSeek OCR PRO V2", font=("Arial", 24, "bold")).pack(side="left", padx=20)

        self.btn_cancel = ctk.CTkButton(top, text="Cancel", command=self._cancel, fg_color="red", state="disabled")
        self.btn_cancel.pack(side="right", padx=10)

        self.btn_select = ctk.CTkButton(
            top, text="Select PDF(s)", command=self._select_pdfs,
            height=40, font=("Arial", 16, "bold"),
        )
        self.btn_select.pack(side="right", padx=20)

        # Options
        opts = ctk.CTkFrame(self.root)
        opts.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.txt_var = ctk.BooleanVar(value=True)
        self.txt_page_var = ctk.BooleanVar(value=False)
        self.docx_var = ctk.BooleanVar(value=False)
        self.md_var = ctk.BooleanVar(value=True)
        self.img_var = ctk.BooleanVar(value=False)
        self.fast_var = ctk.BooleanVar(value=False)
        self.resume_var = ctk.BooleanVar(value=True)
        self.quantize_var = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(opts, text="TXT (Single)", variable=self.txt_var).grid(row=0, column=0, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="TXT Per Page", variable=self.txt_page_var).grid(row=0, column=1, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="DOCX", variable=self.docx_var).grid(row=0, column=2, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="Markdown", variable=self.md_var).grid(row=0, column=3, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="Extract Images", variable=self.img_var).grid(row=1, column=0, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="Fast Mode", variable=self.fast_var).grid(row=1, column=1, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="Resume", variable=self.resume_var).grid(row=1, column=2, padx=15, pady=8, sticky="w")
        ctk.CTkCheckBox(opts, text="INT8 Quantize", variable=self.quantize_var).grid(row=1, column=3, padx=15, pady=8, sticky="w")

        # Progress
        self.status_label = ctk.CTkLabel(self.root, text="Ready", font=("Arial", 14))
        self.status_label.grid(row=2, column=0, pady=(10, 0))

        self.progress = ctk.CTkProgressBar(self.root, width=800)
        self.progress.set(0)
        self.progress.grid(row=3, column=0, pady=(10, 0))

        self.pct_label = ctk.CTkLabel(self.root, text="0%", font=("Arial", 12))
        self.pct_label.grid(row=4, column=0, pady=(0, 10))

        # Logs
        log_frame = ctk.CTkFrame(self.root)
        log_frame.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
        self.root.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(log_frame, text="Console Output").pack(pady=5)
        self.log_text = ctk.CTkTextbox(
            log_frame, fg_color="black", text_color="#00FF00",
            font=("Consolas", 12),
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_config(self, pdf_paths: list[str]) -> OCRConfig:
        formats = []
        if self.txt_var.get():
            formats.append("txt")
        if self.txt_page_var.get():
            formats.append("txt_pages")
        if self.docx_var.get():
            formats.append("docx")
        if self.md_var.get():
            formats.append("markdown")
        if not formats:
            formats = ["txt"]

        model_path = str(resolve_model_path("./models"))

        return OCRConfig(
            pdf_paths=list(pdf_paths),
            model_path=model_path,
            quantize="int8" if self.quantize_var.get() else "none",
            mode="fast" if self.fast_var.get() else "accurate",
            output_dir="./output",
            formats=formats,
            extract_images=self.img_var.get(),
            resume=self.resume_var.get(),
        )

    def _select_pdfs(self):
        if self._is_processing:
            return
        files = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if not files:
            return

        self._set_busy(True)
        config = self._build_config(files)

        errors = config.validate()
        if errors:
            messagebox.showerror("Configuration Error", "\n".join(errors))
            self._set_busy(False)
            return

        self._cancel_event.clear()
        threading.Thread(target=self._run_ocr, args=(config,), daemon=True).start()

    def _run_ocr(self, config: OCRConfig):
        from ..engine.ocr_engine import OCREngine
        from ..pipeline.orchestrator import Orchestrator

        try:
            if self._engine is None:
                self._log("Loading model (this may take a few minutes)...")
                self._set_status("Loading AI model...")
                self._engine = OCREngine(config)
                self._engine.load_model()
                self._log("Model loaded successfully")

            orchestrator = Orchestrator(
                config, self._engine,
                progress_callback=self._on_progress,
                status_callback=self._set_status,
                cancel_check=lambda: self._cancel_event.is_set(),
            )
            orchestrator.run_all()

            self._log("All tasks completed!")
            self._set_status("Complete")

        except Exception as e:
            self._log(f"Error: {e}")
            self._set_status(f"Error: {e}")

        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _cancel(self):
        self._cancel_event.set()
        self._set_status("Cancelling...")

    def _set_busy(self, busy: bool):
        self._is_processing = busy
        state = "disabled" if busy else "normal"
        self.btn_select.configure(state=state)
        self.btn_cancel.configure(state="normal" if busy else "disabled")

    def _log(self, msg: str):
        self.root.after(0, self._append_log, str(msg))

    def _append_log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _set_status(self, text: str):
        self.root.after(0, lambda: self.status_label.configure(text=text))

    def _on_progress(self, current: int, total: int, eta: float):
        def _update():
            pct = current / total if total > 0 else 0
            self.progress.set(pct)
            self.pct_label.configure(text=f"{int(pct * 100)}%")
        self.root.after(0, _update)


def main():
    """Entry point for gui-scripts."""
    launch_gui()
