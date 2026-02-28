"""Base utilities for output writers."""

from __future__ import annotations

import logging
from pathlib import Path

from ..engine.ocr_engine import PageResult
from ..engine.text_postprocessor import clean_ocr_text

log = logging.getLogger(__name__)


def get_texts_from_results(results: list[PageResult], clean: bool = True) -> list[str]:
    """Extract text from page results, optionally cleaning them."""
    texts = []
    for r in results:
        if r.error:
            texts.append(f"[Error on page {r.page_number + 1}: {r.error}]")
        elif clean:
            texts.append(clean_ocr_text(r.raw_text))
        else:
            texts.append(r.raw_text)
    return texts


def safe_write(filepath: str, content: str) -> str:
    """Write content to file, handling permission errors with numbered fallback.

    Returns the actual path written to.
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        p.write_text(content, encoding="utf-8")
        return str(p)
    except PermissionError:
        # Try numbered alternatives
        for i in range(1, 100):
            alt = p.parent / f"{p.stem}_{i}{p.suffix}"
            try:
                alt.write_text(content, encoding="utf-8")
                log.warning(f"Permission denied on {p.name}, wrote to {alt.name}")
                return str(alt)
            except PermissionError:
                continue
        raise PermissionError(f"Cannot write to {filepath} or any alternative")
