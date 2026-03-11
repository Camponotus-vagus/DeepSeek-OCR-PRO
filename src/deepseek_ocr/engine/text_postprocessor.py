"""Post-processing of model output text."""

from __future__ import annotations

import re
import logging
import json

log = logging.getLogger(__name__)


def clean_ocr_text(raw_text: str) -> str:
    """Clean raw model output for plain text usage.

    Removes grounding tags, normalizes whitespace, and fixes common artifacts.
    """
    text = raw_text

    # Remove grounding reference tags: <|ref|>...<|/ref|><|det|>...<|/det|>
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)

    # Remove any remaining special tokens
    text = re.sub(r"<\|[^|]+\|>", "", text)

    # Fix LaTeX artifacts
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

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
            coords_list = json.loads(coords_str)
            for coords in coords_list:
                if len(coords) == 4:
                    regions.append({
                        "label": label.strip(),
                        "coords": coords,  # [x1, y1, x2, y2] normalized to 0-999
                    })
        except Exception as e:
            log.debug(f"Failed to parse grounding coords '{coords_str}': {e}")

    return regions
