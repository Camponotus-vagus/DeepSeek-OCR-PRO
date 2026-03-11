"""PDF handling: page extraction and metadata using PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


class PDFHandler:
    """Extract page images from PDFs using PyMuPDF (fitz)."""

    def __init__(self, fallback_dpi: int = 200):
        self._fallback_dpi = fallback_dpi

    def get_page_count(self, pdf_path: str) -> int:
        """Return the number of pages in a PDF."""
        import fitz

        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            log.error(f"Failed to read PDF {pdf_path}: {e}")
            return 0

    def extract_page_image(self, pdf_path: str, page_num: int) -> Image.Image:
        """Extract a page as a PIL Image.

        For scanned PDFs (single embedded image per page), extracts the original
        image directly to preserve quality. Falls back to rasterization for
        complex pages.

        Args:
            pdf_path: Path to the PDF file.
            page_num: 0-indexed page number.

        Returns:
            PIL Image in RGB mode.

        Raises:
            RuntimeError: If the page cannot be extracted.
        """
        import fitz

        doc = fitz.open(pdf_path)
        try:
            if page_num < 0 or page_num >= len(doc):
                raise RuntimeError(f"Page {page_num} out of range (0-{len(doc)-1})")

            page = doc[page_num]

            # Strategy 1: Direct embedded image extraction (best for scanned PDFs)
            img = self._try_extract_embedded(doc, page)
            if img is not None:
                return img

            # Strategy 2: Rasterize the page
            return self._rasterize_page(page)
        finally:
            doc.close()

    def _try_extract_embedded(self, doc, page) -> Image.Image | None:
        """Try to extract a single embedded image from a page.

        This works for scanned PDFs where each page is one big image.
        Returns None if the page has multiple images or no images.
        """
        try:
            images = page.get_images(full=True)
            if len(images) != 1:
                return None

            xref = images[0][0]
            base_image = doc.extract_image(xref)

            if base_image is None:
                return None

            image_bytes = base_image["image"]
            import io
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if img.mode == "1":
                # 1-bit B&W: convert to grayscale first for smoother edges
                img = img.convert("L")
            if img.mode != "RGB":
                img = img.convert("RGB")

            log.debug(f"Extracted embedded image: {img.size[0]}x{img.size[1]}")
            return img

        except Exception as e:
            log.debug(f"Embedded extraction failed, will rasterize: {e}")
            return None

    def _rasterize_page(self, page) -> Image.Image:
        """Rasterize a page at the configured DPI."""
        import fitz

        dpi = self._fallback_dpi
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        log.debug(f"Rasterized page at {dpi} DPI: {img.size[0]}x{img.size[1]}")
        return img

    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get basic PDF metadata."""
        import fitz

        doc = fitz.open(pdf_path)
        info = {
            "path": str(Path(pdf_path).resolve()),
            "filename": Path(pdf_path).name,
            "page_count": len(doc),
            "metadata": doc.metadata,
        }
        if len(doc) > 0:
            page = doc[0]
            rect = page.rect
            info["page_size"] = (rect.width, rect.height)
            images = page.get_images(full=True)
            info["is_scanned"] = len(images) == 1
        doc.close()
        return info
