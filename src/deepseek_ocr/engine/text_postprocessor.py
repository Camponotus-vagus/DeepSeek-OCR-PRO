"""Post-processing of model output text."""

from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repetition & garbage detection helpers
# ---------------------------------------------------------------------------

def _collapse_repeated_lines(text: str, max_repeats: int = 3) -> str:
    """Collapse consecutive identical lines down to *max_repeats* copies."""
    lines = text.split("\n")
    out: list[str] = []
    prev = None
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev:
            count += 1
            if count <= max_repeats:
                out.append(line)
        else:
            prev = stripped
            count = 1
            out.append(line)
    return "\n".join(out)


def _remove_numeric_floods(text: str) -> str:
    r"""Remove lines that are *only* long numeric / dot sequences.

    Catches patterns like:
      ``0.0.0.0.0.0.0.0.0.0...``
      ``1.1.1.1.1.1.2.1.1...``
      ``20 21 22 23 24 25 26 27...``
      ``text[[0.0, 0.0, 997, 0.0, ...]]``
    """
    cleaned: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        # Skip empty lines (preserve them)
        if not s:
            cleaned.append(line)
            continue
        # Pattern 1: dot-separated digit floods  e.g. "0.0.0.0.0..."
        if re.fullmatch(r"[\d]+(?:\.[\d]+){10,}", s):
            log.debug("Removed numeric flood: %s...", s[:60])
            continue
        # Pattern 2: space-separated integer floods  e.g. "20 21 22 23 ..."
        if re.fullmatch(r"\d+(?:\s+\d+){15,}", s):
            log.debug("Removed integer sequence: %s...", s[:60])
            continue
        # Pattern 3: coordinate/array floods  e.g. "text[[0.0, 0.0, 997, ..."
        if re.search(r"\[\[[\d.,\s]{100,}", s):
            log.debug("Removed coordinate flood: %s...", s[:60])
            continue
        # Pattern 4: "None." floods in table cells
        if s.count("None") > 10:
            log.debug("Removed None flood: %s...", s[:60])
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _remove_empty_html_tables(text: str) -> str:
    """Remove HTML <table> blocks that contain mostly empty cells or 'None'."""
    def _is_junk_table(match: re.Match) -> str:
        table_html = match.group(0)
        # Count real content vs empty/None cells
        cells = re.findall(r"<td[^>]*>(.*?)</td>", table_html, re.DOTALL)
        if not cells:
            return table_html
        non_empty = sum(1 for c in cells if c.strip() and c.strip() not in ("None", "None."))
        if len(cells) > 5 and non_empty / len(cells) < 0.15:
            log.debug("Removed junk table with %d/%d empty cells", len(cells) - non_empty, len(cells))
            return ""
        return table_html

    return re.sub(r"<table>.*?</table>", _is_junk_table, text, flags=re.DOTALL)


def clean_ocr_text(raw_text: str) -> str:
    """Clean raw model output for plain text usage.

    Removes grounding tags, normalizes whitespace, fixes common artifacts,
    and strips repetitive / garbage output from degenerate model runs.
    """
    text = raw_text

    # Remove grounding reference tags: <|ref|>...<|/ref|><|det|>...<|/det|>
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)

    # Remove any remaining special tokens
    text = re.sub(r"<\|[^|]+\|>", "", text)

    # Fix LaTeX artifacts
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

    # --- Garbage / repetition cleanup ---
    text = _remove_numeric_floods(text)
    text = _remove_empty_html_tables(text)
    text = _collapse_repeated_lines(text)

    # Normalize multiple blank lines to at most 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def clean_for_markdown(raw_text: str) -> str:
    """Clean raw model output while preserving markdown formatting.

    Replaces image grounding tags with markdown image references.
    """
    text = raw_text

    # Replace image references with markdown format
    # <|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    img_idx = 0

    def replace_image_ref(match):
        nonlocal img_idx
        ref_content = match.group(1)
        if "image" in ref_content.lower():
            idx = img_idx
            img_idx += 1
            return f"![image_{idx}](images/image_{idx}.png)"
        # Non-image references: just remove the tags
        return ref_content

    text = re.sub(
        r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>",
        replace_image_ref,
        text,
        flags=re.DOTALL,
    )

    # Remove any remaining special tokens
    text = re.sub(r"<\|[^|]+\|>", "", text)

    # Fix LaTeX artifacts
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

    # --- Garbage / repetition cleanup ---
    text = _remove_numeric_floods(text)
    text = _remove_empty_html_tables(text)
    text = _collapse_repeated_lines(text)

    # Normalize multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def extract_grounding_regions(raw_text: str) -> list[dict]:
    """Extract image region coordinates from grounding tags.

    Returns list of dicts with 'label', 'coords' (list of [x1,y1,x2,y2] normalized 0-999).
    """
    pattern = r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    regions = []
    for label, coords_str in matches:
        if "image" not in label.lower():
            continue
        try:
            # coords_str is like "[[x1,y1,x2,y2]]" or "[[x1,y1,x2,y2],[...]]"
            coords_list = eval(coords_str)  # Safe here: model output, not user input
            for coords in coords_list:
                if len(coords) == 4:
                    regions.append({
                        "label": label.strip(),
                        "coords": coords,  # [x1, y1, x2, y2] normalized to 0-999
                    })
        except Exception as e:
            log.debug(f"Failed to parse grounding coords '{coords_str}': {e}")

    return regions
