"""TXT output writer."""

from __future__ import annotations

import logging
from pathlib import Path

from ..engine.ocr_engine import PageResult
from .writer_base import get_texts_from_results, safe_write

log = logging.getLogger(__name__)

PAGE_SEPARATOR = "\n\n" + "=" * 80 + "\n\n"


def write_txt(results: list[PageResult], output_dir: str, pdf_stem: str) -> str:
    """Write all pages to a single TXT file.

    Returns the path of the written file.
    """
    texts = get_texts_from_results(results)
    content = PAGE_SEPARATOR.join(texts)

    filepath = str(Path(output_dir) / f"{pdf_stem}.txt")
    actual = safe_write(filepath, content)
    log.info(f"TXT written: {actual}")
    return actual


def write_txt_per_page(results: list[PageResult], output_dir: str, pdf_stem: str) -> list[str]:
    """Write each page to a separate TXT file.

    Returns list of written file paths.
    """
    texts = get_texts_from_results(results)
    written = []

    pages_dir = Path(output_dir) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(texts):
        filepath = str(pages_dir / f"{pdf_stem}_page_{i + 1:03d}.txt")
        actual = safe_write(filepath, text)
        written.append(actual)

    log.info(f"TXT per page: {len(written)} files written")
    return written
