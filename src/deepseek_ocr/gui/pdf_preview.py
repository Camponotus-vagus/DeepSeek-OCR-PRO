"""PDF preview widget using PyMuPDF."""

from __future__ import annotations

import logging
import threading
import tkinter as tk

log = logging.getLogger(__name__)


class PDFPreview(tk.Frame):
    """Embeddable PDF page preview widget."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self._doc = None
        self._current_page = 0
        self._total_pages = 0
        self._zoom = 1.0
        self._photo = None

        # Canvas
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Navigation bar
        nav = tk.Frame(self)
        nav.pack(fill="x", pady=5)

        self.btn_prev = tk.Button(nav, text="< Prev", command=self._prev_page)
        self.btn_prev.pack(side="left", padx=5)

        self.page_label = tk.Label(nav, text="No PDF loaded")
        self.page_label.pack(side="left", expand=True)

        self.btn_next = tk.Button(nav, text="Next >", command=self._next_page)
        self.btn_next.pack(side="right", padx=5)

    def load_pdf(self, pdf_path: str) -> None:
        """Load a PDF file for preview."""
        threading.Thread(target=self._load_async, args=(pdf_path,), daemon=True).start()

    def _load_async(self, pdf_path: str) -> None:
        try:
            import fitz
            if self._doc:
                self._doc.close()
            self._doc = fitz.open(pdf_path)
            self._total_pages = len(self._doc)
            self._current_page = 0
            self.after(0, self._render_current)
        except Exception as e:
            log.error(f"Failed to load PDF for preview: {e}")
            err_msg = str(e)
            self.after(0, lambda: self.page_label.configure(text=f"Error: {err_msg}"))

    def _render_current(self) -> None:
        if not self._doc or self._total_pages == 0:
            return

        try:
            from PIL import Image, ImageTk
            import fitz as fitz_mod

            page = self._doc[self._current_page]
            mat = fitz_mod.Matrix(self._zoom, self._zoom)
            pix = page.get_pixmap(matrix=mat)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Scale to fit canvas
            cw = self.canvas.winfo_width() or 400
            ch = self.canvas.winfo_height() or 600
            ratio = min(cw / img.width, ch / img.height, 1.0)
            if ratio < 1.0:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

            self.page_label.configure(
                text=f"Page {self._current_page + 1} / {self._total_pages}"
            )
        except Exception as e:
            log.error(f"Failed to render page: {e}")

    def _prev_page(self) -> None:
        if self._current_page > 0:
            self._current_page -= 1
            self._render_current()

    def _next_page(self) -> None:
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._render_current()
