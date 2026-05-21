"""Tests for the OCR engine (marker-pdf integration mocked)."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from surya_ocr.config import OCRConfig
from surya_ocr.engine.ocr_engine import OCREngine, PageResult


# --------- marker-pdf fakes installed into sys.modules ---------

def _install_fake_marker(monkeypatch, *, markdown_output="page text", raise_on_convert=False):
    """Install fake marker.* modules. Returns a handle exposing fakes for assertion."""
    handle = types.SimpleNamespace()

    # marker.models.create_model_dict
    fake_models = types.ModuleType("marker.models")
    handle.model_dict = {"layout": object()}
    fake_models.create_model_dict = MagicMock(return_value=handle.model_dict)

    # marker.config.parser.ConfigParser
    fake_config_parser_module = types.ModuleType("marker.config.parser")
    handle.config_parser_instance = MagicMock()
    handle.config_parser_instance.generate_config_dict.return_value = {"k": "v"}
    fake_config_parser_module.ConfigParser = MagicMock(return_value=handle.config_parser_instance)
    handle.ConfigParser = fake_config_parser_module.ConfigParser

    # marker.converters.pdf.PdfConverter
    fake_converters_pdf = types.ModuleType("marker.converters.pdf")
    rendered = MagicMock()
    rendered.markdown = markdown_output

    converter_instance = MagicMock()
    if raise_on_convert:
        converter_instance.side_effect = RuntimeError("synthetic OCR failure")
    else:
        converter_instance.return_value = rendered

    fake_converters_pdf.PdfConverter = MagicMock(return_value=converter_instance)
    handle.PdfConverter = fake_converters_pdf.PdfConverter
    handle.converter_instance = converter_instance

    # Build parent namespaces too so `import marker.X.Y` works
    fake_marker = types.ModuleType("marker")
    fake_marker_config = types.ModuleType("marker.config")
    fake_marker_converters = types.ModuleType("marker.converters")

    monkeypatch.setitem(sys.modules, "marker", fake_marker)
    monkeypatch.setitem(sys.modules, "marker.config", fake_marker_config)
    monkeypatch.setitem(sys.modules, "marker.config.parser", fake_config_parser_module)
    monkeypatch.setitem(sys.modules, "marker.converters", fake_marker_converters)
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", fake_converters_pdf)
    monkeypatch.setitem(sys.modules, "marker.models", fake_models)

    return handle


@pytest.fixture
def cfg(sample_pdf, tmp_output):
    return OCRConfig(
        pdf_paths=[sample_pdf],
        output_dir=tmp_output,
        languages=["it", "la"],
        force_ocr=True,
    )


# ---------- PageResult ----------

class TestPageResult:
    def test_round_trip(self):
        r = PageResult(page_number=3, raw_text="hello", processing_time=1.5)
        restored = PageResult.from_dict(r.to_dict())
        assert restored == r

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            "page_number": 1,
            "raw_text": "x",
            "processing_time": 0.0,
            "error": None,
            "extra_field": "ignored",
        }
        r = PageResult.from_dict(data)
        assert r.page_number == 1
        assert r.error is None

    def test_with_error(self):
        r = PageResult(page_number=0, raw_text="", processing_time=0.0, error="boom")
        restored = PageResult.from_dict(r.to_dict())
        assert restored.error == "boom"


# ---------- load_model ----------

class TestLoadModel:
    def test_loads_and_marks_ready(self, monkeypatch, cfg):
        marker = _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        engine.load_model()

        assert engine.is_loaded
        marker.ConfigParser.assert_called_once()
        # ConfigParser was called with the config we built from cfg
        passed_cfg = marker.ConfigParser.call_args[0][0]
        assert passed_cfg["force_ocr"] is True
        assert passed_cfg["languages"] == ["it", "la"]
        assert passed_cfg["output_format"] == "markdown"

    def test_idempotent(self, monkeypatch, cfg):
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        engine.load_model()
        engine.load_model()
        # create_model_dict only called once on second load
        assert sys.modules["marker.models"].create_model_dict.call_count == 1


# ---------- process_pdf ----------

class TestProcessPdf:
    def test_raises_when_not_loaded(self, cfg):
        engine = OCREngine(cfg)
        with pytest.raises(RuntimeError, match="Models not loaded"):
            engine.process_pdf("anything.pdf")

    def test_happy_path_returns_page_results(self, monkeypatch, cfg, sample_pdf):
        marker = _install_fake_marker(monkeypatch, markdown_output="hello world")
        engine = OCREngine(cfg)
        engine.load_model()

        results = engine.process_pdf(sample_pdf)
        assert len(results) >= 1
        assert results[0].raw_text == "hello world"
        assert results[0].error is None
        assert results[0].processing_time >= 0
        marker.PdfConverter.assert_called_once()

    def test_converter_exception_returns_error_result(self, monkeypatch, cfg, sample_pdf):
        _install_fake_marker(monkeypatch, raise_on_convert=True)
        engine = OCREngine(cfg)
        engine.load_model()

        results = engine.process_pdf(sample_pdf)
        assert len(results) == 1
        assert results[0].error is not None
        assert "synthetic OCR failure" in results[0].error
        assert results[0].raw_text == ""


# ---------- process_pdf_by_page ----------

class TestProcessPdfByPage:
    def test_raises_when_not_loaded(self, cfg):
        engine = OCREngine(cfg)
        with pytest.raises(RuntimeError, match="Models not loaded"):
            engine.process_pdf_by_page("anything.pdf", 0)

    def test_happy_path(self, monkeypatch, cfg, multi_page_pdf):
        _install_fake_marker(monkeypatch, markdown_output="page 2 text")
        engine = OCREngine(cfg)
        engine.load_model()

        result = engine.process_pdf_by_page(multi_page_pdf, page_num=1)
        assert result.page_number == 1
        assert result.raw_text == "page 2 text"
        assert result.error is None

    def test_temp_dir_is_cleaned(self, monkeypatch, cfg, sample_pdf):
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        engine.load_model()

        captured = {}
        import tempfile as _tempfile

        real_mkdtemp = _tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            captured["dir"] = d
            return d

        with patch.object(_tempfile, "mkdtemp", tracking_mkdtemp):
            engine.process_pdf_by_page(sample_pdf, 0)

        assert "dir" in captured
        assert not os.path.exists(captured["dir"]), "Temp directory should be cleaned up"

    def test_exception_returns_error_result(self, monkeypatch, cfg, sample_pdf):
        _install_fake_marker(monkeypatch, raise_on_convert=True)
        engine = OCREngine(cfg)
        engine.load_model()

        result = engine.process_pdf_by_page(sample_pdf, 0)
        assert result.error is not None
        assert "synthetic OCR failure" in result.error
        assert result.page_number == 0
        assert result.raw_text == ""


# ---------- _split_into_pages ----------

class TestSplitIntoPages:
    def test_single_page_pdf(self, monkeypatch, cfg, sample_pdf):
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        results = engine._split_into_pages("the whole text", sample_pdf)
        assert len(results) == 1
        assert results[0].raw_text == "the whole text"
        assert results[0].page_number == 0

    def test_multi_page_pdf_returns_one_result(self, monkeypatch, cfg, multi_page_pdf):
        # Current behavior: marker-pdf output is treated as one continuous result
        # regardless of page count. This test locks that contract in.
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        results = engine._split_into_pages("entire doc", multi_page_pdf)
        assert len(results) == 1
        assert results[0].raw_text == "entire doc"


# ---------- unload_model ----------

class TestUnloadModel:
    def test_no_op_when_not_loaded(self, cfg):
        engine = OCREngine(cfg)
        engine.unload_model()  # should not raise
        assert not engine.is_loaded

    def test_clears_state_and_collects(self, monkeypatch, cfg):
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        engine.load_model()
        assert engine.is_loaded

        with patch("gc.collect") as mock_gc:
            engine.unload_model()

        assert not engine.is_loaded
        assert engine._model_dict is None
        mock_gc.assert_called_once()

    def test_empties_cuda_cache_when_available(self, monkeypatch, cfg):
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        engine.load_model()

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=MagicMock(return_value=True),
            empty_cache=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        engine.unload_model()
        fake_torch.cuda.empty_cache.assert_called_once()

    def test_skips_cuda_when_unavailable(self, monkeypatch, cfg):
        _install_fake_marker(monkeypatch)
        engine = OCREngine(cfg)
        engine.load_model()

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=MagicMock(return_value=False),
            empty_cache=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        engine.unload_model()
        fake_torch.cuda.empty_cache.assert_not_called()
