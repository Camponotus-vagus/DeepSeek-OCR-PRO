"""Markdown output writer."""

from __future__ import annotations

import logging
from pathlib import Path

from ..engine.ocr_engine import PageResult
from ..engine.text_postprocessor import clean_for_markdown
from .writer_base import safe_write

log = logging.getLogger(__name__)


def write_markdown(results: list[PageResult], output_dir: str, pdf_stem: str) -> str:
    """Write all pages to a Markdown file preserving model formatting.

    The DeepSeek model outputs markdown when using the layout prompt.
    This writer preserves that formatting and adds page separators.

    Returns the path of the written file.
    """
    pages = []

    for r in results:
        if r.error:
            pages.append(f"> **Error on page {r.page_number + 1}:** {r.error}")
        else:
            cleaned = clean_for_markdown(r.raw_text)
            pages.append(cleaned)

    # Join pages with markdown horizontal rules
    separator = "\n\n---\n\n"
    content = separator.join(pages)

    # Add document header
    header = f"# {pdf_stem}\n\n"
    content = header + content

    filepath = str(Path(output_dir) / f"{pdf_stem}.md")
    actual = safe_write(filepath, content)
    log.info(f"Markdown written: {actual}")
    return actual
