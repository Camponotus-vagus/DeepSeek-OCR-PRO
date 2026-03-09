"""Tests for configuration module."""

import json
import pytest
from pathlib import Path

from deepseek_ocr.config import OCRConfig


class TestOCRConfig:
    def test_defaults(self):
        config = OCRConfig()
        assert config.mode == "accurate"
        assert config.quantize == "int8"
        assert config.device == "auto"
        assert config.formats == ["txt"]
        assert config.resume is False
        assert config.extract_images is False
        assert config.max_new_tokens == 4096

    def test_base_size_accurate(self):
        config = OCRConfig(mode="accurate")
        assert config.base_size == 1024
        assert config.crop_mode is True

    def test_base_size_fast(self):
        config = OCRConfig(mode="fast")
        assert config.base_size == 640
        assert config.crop_mode is False

    def test_prompt_layout(self):
        config = OCRConfig(prompt_mode="layout")
        assert "<|grounding|>" in config.prompt
        assert "markdown" in config.prompt.lower()

    def test_prompt_freeocr(self):
        config = OCRConfig(prompt_mode="freeocr")
        assert "Free OCR" in config.prompt

    def test_validate_no_pdfs(self):
        config = OCRConfig()
        errors = config.validate()
        assert any("No PDF" in e for e in errors)

    def test_validate_missing_pdf(self):
        config = OCRConfig(pdf_paths=["/nonexistent/file.pdf"])
        errors = config.validate()
        assert any("not found" in e for e in errors)

    def test_validate_not_a_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        config = OCRConfig(pdf_paths=[str(txt_file)])
        errors = config.validate()
        assert any("Not a PDF" in e for e in errors)

    def test_validate_invalid_mode(self):
        config = OCRConfig(pdf_paths=[], mode="invalid")
        errors = config.validate()
        assert any("Invalid mode" in e for e in errors)

    def test_validate_invalid_format(self):
        config = OCRConfig(pdf_paths=[], formats=["html"])
        errors = config.validate()
        assert any("Invalid format" in e for e in errors)

    def test_validate_invalid_quantize(self):
        config = OCRConfig(pdf_paths=[], quantize="int4")
        errors = config.validate()
        assert any("Invalid quantize" in e for e in errors)

    def test_validate_invalid_workers(self):
        config = OCRConfig(pdf_paths=[], num_workers=0)
        errors = config.validate()
        assert any("num_workers" in e for e in errors)

    def test_validate_invalid_max_new_tokens(self):
        config = OCRConfig(pdf_paths=[], max_new_tokens=50)
        errors = config.validate()
        assert any("max_new_tokens" in e for e in errors)

    def test_max_new_tokens_custom(self):
        config = OCRConfig(max_new_tokens=2048)
        assert config.max_new_tokens == 2048

    def test_from_file_json(self, tmp_path):
        config_data = {
            "mode": "fast",
            "quantize": "none",
            "formats": ["txt", "markdown"],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = OCRConfig.from_file(str(config_file))
        assert config.mode == "fast"
        assert config.quantize == "none"
        assert config.formats == ["txt", "markdown"]

    def test_to_dict(self):
        config = OCRConfig(mode="fast", formats=["txt", "docx"])
        d = config.to_dict()
        assert d["mode"] == "fast"
        assert d["formats"] == ["txt", "docx"]
        assert isinstance(d, dict)

    def test_valid_config(self, sample_pdf, tmp_path):
        config = OCRConfig(
            pdf_paths=[sample_pdf],
            model_path=str(tmp_path),  # Won't have config.json but tests other validation
        )
        errors = config.validate()
        # Only model path error expected (no config.json in tmp_path)
        assert all("Model config.json" in e for e in errors)
