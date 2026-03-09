"""Tests for text post-processing."""

from deepseek_ocr.engine.text_postprocessor import (
    clean_ocr_text,
    clean_for_markdown,
    extract_grounding_regions,
)


class TestCleanOcrText:
    def test_basic_text(self):
        assert clean_ocr_text("Hello world") == "Hello world"

    def test_removes_grounding_tags(self):
        text = "Some text <|ref|>image<|/ref|><|det|>[[100,100,200,200]]<|/det|> more text"
        result = clean_ocr_text(text)
        assert "<|ref|>" not in result
        assert "<|det|>" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_removes_special_tokens(self):
        text = "Hello <|special|> world"
        assert "<|special|>" not in clean_ocr_text(text)

    def test_fixes_latex(self):
        text = "a \\coloneqq b and c \\eqqcolon d"
        result = clean_ocr_text(text)
        assert ":=" in result
        assert "=:" in result

    def test_normalizes_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = clean_ocr_text(text)
        assert result == "line1\n\nline2"

    def test_strips_trailing_whitespace(self):
        text = "line1   \nline2  \n"
        result = clean_ocr_text(text)
        assert "   " not in result

    def test_empty_text(self):
        assert clean_ocr_text("") == ""

    def test_only_tags(self):
        text = "<|ref|>image<|/ref|><|det|>[[0,0,100,100]]<|/det|>"
        result = clean_ocr_text(text)
        assert result == ""

    def test_collapses_repeated_lines(self):
        text = "Hello\n" * 20
        result = clean_ocr_text(text)
        assert result.count("Hello") == 3

    def test_removes_numeric_dot_floods(self):
        text = "Good text\n0.0.0.0.0.0.0.0.0.0.0.0.0.0.0\nMore good text"
        result = clean_ocr_text(text)
        assert "0.0.0.0" not in result
        assert "Good text" in result
        assert "More good text" in result

    def test_removes_integer_sequence_floods(self):
        nums = " ".join(str(i) for i in range(1, 100))
        text = f"Title\n{nums}\nContent"
        result = clean_ocr_text(text)
        assert "50 51 52" not in result
        assert "Title" in result
        assert "Content" in result

    def test_removes_coordinate_floods(self):
        text = "text[[0.0, 0.0, 997, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5]]"
        result = clean_ocr_text(text)
        assert "[[0.0" not in result

    def test_removes_junk_html_tables(self):
        text = '<table><tr><td></td><td></td><td>None</td><td>None</td><td>None</td><td>None</td><td>None</td><td>None</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>'
        result = clean_ocr_text(text)
        assert "<table>" not in result

    def test_preserves_real_tables(self):
        text = '<table><tr><td>Name</td><td>Value</td></tr><tr><td>Alpha</td><td>100</td></tr></table>'
        result = clean_ocr_text(text)
        assert "<table>" in result
        assert "Alpha" in result


class TestCleanForMarkdown:
    def test_basic_text(self):
        assert clean_for_markdown("# Title\n\nParagraph") == "# Title\n\nParagraph"

    def test_replaces_image_refs(self):
        text = "Text <|ref|>image<|/ref|><|det|>[[100,100,200,200]]<|/det|> more"
        result = clean_for_markdown(text)
        assert "![image_0]" in result
        assert "images/image_0.png" in result

    def test_multiple_image_refs(self):
        text = (
            "A <|ref|>image<|/ref|><|det|>[[0,0,100,100]]<|/det|> "
            "B <|ref|>image<|/ref|><|det|>[[200,200,300,300]]<|/det|>"
        )
        result = clean_for_markdown(text)
        assert "image_0" in result
        assert "image_1" in result

    def test_non_image_refs_removed(self):
        text = "<|ref|>table<|/ref|><|det|>[[0,0,100,100]]<|/det|>"
        result = clean_for_markdown(text)
        assert "table" in result
        assert "<|ref|>" not in result


class TestExtractGroundingRegions:
    def test_extract_image_regions(self):
        text = "<|ref|>image<|/ref|><|det|>[[100,200,300,400]]<|/det|>"
        regions = extract_grounding_regions(text)
        assert len(regions) == 1
        assert regions[0]["coords"] == [100, 200, 300, 400]

    def test_extract_multiple_regions(self):
        text = (
            "<|ref|>image<|/ref|><|det|>[[0,0,100,100]]<|/det|>"
            "<|ref|>image<|/ref|><|det|>[[500,500,999,999]]<|/det|>"
        )
        regions = extract_grounding_regions(text)
        assert len(regions) == 2

    def test_ignores_non_image_refs(self):
        text = "<|ref|>table<|/ref|><|det|>[[0,0,100,100]]<|/det|>"
        regions = extract_grounding_regions(text)
        assert len(regions) == 0

    def test_no_regions(self):
        assert extract_grounding_regions("plain text") == []

    def test_malformed_coords(self):
        text = "<|ref|>image<|/ref|><|det|>invalid<|/det|>"
        regions = extract_grounding_regions(text)
        assert len(regions) == 0
