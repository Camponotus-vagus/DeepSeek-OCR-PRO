"""Tests for output writer base utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from surya_ocr.engine.ocr_engine import PageResult
from surya_ocr.output.writer_base import get_texts_from_results, safe_write


class TestGetTextsFromResults:
    def test_clean_strips_markdown(self):
        results = [
            PageResult(page_number=0, raw_text="## Header\n\nbody", processing_time=1.0),
        ]
        texts = get_texts_from_results(results, clean=True)
        assert len(texts) == 1
        assert "##" not in texts[0]
        assert "Header" in texts[0]
        assert "body" in texts[0]

    def test_no_clean_preserves_raw(self):
        raw = "## Header\n\n**bold**\n"
        results = [PageResult(page_number=0, raw_text=raw, processing_time=1.0)]
        texts = get_texts_from_results(results, clean=False)
        assert texts == [raw]

    def test_error_pages_get_placeholder(self):
        results = [
            PageResult(page_number=0, raw_text="ok", processing_time=1.0),
            PageResult(page_number=4, raw_text="", processing_time=0.0, error="boom"),
        ]
        texts = get_texts_from_results(results)
        assert texts[0] == "ok"
        # page_number 4 → human-readable "page 5"
        assert "Error on page 5" in texts[1]
        assert "boom" in texts[1]

    def test_empty_input(self):
        assert get_texts_from_results([]) == []


class TestSafeWrite:
    def test_writes_to_target(self, tmp_path):
        target = tmp_path / "out.txt"
        written = safe_write(str(target), "hello")
        assert written == str(target)
        assert target.read_text(encoding="utf-8") == "hello"

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "out.txt"
        safe_write(str(target), "x")
        assert target.exists()

    def test_unicode_content(self, tmp_path):
        target = tmp_path / "u.txt"
        safe_write(str(target), "Carabus italicus à è ù")
        assert "italicus" in target.read_text(encoding="utf-8")

    def test_falls_back_on_permission_error(self, tmp_path):
        target = tmp_path / "locked.txt"
        target.write_text("existing")  # ensure parent exists

        real_write = Path.write_text
        call_count = {"n": 0}

        def flaky_write(self, content, *args, **kwargs):
            # First call (to the original path) fails; subsequent succeed
            call_count["n"] += 1
            if call_count["n"] == 1 and self == target:
                raise PermissionError("locked")
            return real_write(self, content, *args, **kwargs)

        with patch.object(Path, "write_text", flaky_write):
            written = safe_write(str(target), "new content")

        assert written != str(target)
        # Fallback follows the "{stem}_{i}{suffix}" pattern
        assert Path(written).name == "locked_1.txt"
        assert Path(written).read_text(encoding="utf-8") == "new content"

    def test_raises_when_all_fallbacks_fail(self, tmp_path):
        target = tmp_path / "doomed.txt"

        def always_fail(self, content, *args, **kwargs):
            raise PermissionError("nope")

        with patch.object(Path, "write_text", always_fail):
            with pytest.raises(PermissionError):
                safe_write(str(target), "x")
