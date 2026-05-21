"""Integration tests for cli.main() (engine and orchestrator mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from surya_ocr import cli


@pytest.fixture
def _patched_runtime():
    """Patch the OCR engine + orchestrator constructed inside cli._run_ocr."""
    with patch("surya_ocr.engine.ocr_engine.OCREngine") as engine_cls, \
         patch("surya_ocr.pipeline.orchestrator.Orchestrator") as orch_cls:
        engine_instance = MagicMock()
        engine_cls.return_value = engine_instance
        orch_instance = MagicMock()
        orch_cls.return_value = orch_instance
        yield engine_cls, engine_instance, orch_cls, orch_instance


class TestCliMain:
    def test_single_pdf_runs_pipeline(self, sample_pdf, tmp_output, _patched_runtime):
        engine_cls, engine, orch_cls, orch = _patched_runtime

        rc = cli.main([sample_pdf, "-o", tmp_output])
        assert rc == 0

        engine.load_model.assert_called_once()
        orch.run_all.assert_called_once()
        engine.unload_model.assert_called_once()

        # The config passed to Orchestrator has our single PDF
        cfg = orch_cls.call_args[0][0]
        assert cfg.pdf_paths == [sample_pdf]
        assert cfg.output_dir == tmp_output
        assert cfg.formats == ["txt"]  # default

    def test_directory_input_globs_pdfs(self, tmp_path, sample_pdf, multi_page_pdf, _patched_runtime):
        # Drop both fixtures into a single directory
        import shutil
        d = tmp_path / "pdfs"
        d.mkdir()
        a = d / "a.pdf"
        b = d / "b.pdf"
        shutil.copy(sample_pdf, a)
        shutil.copy(multi_page_pdf, b)

        _, _, orch_cls, _ = _patched_runtime
        rc = cli.main([str(d), "-o", str(tmp_path / "out")])
        assert rc == 0

        cfg = orch_cls.call_args[0][0]
        assert sorted(cfg.pdf_paths) == sorted([str(a), str(b)])

    def test_resume_flag_propagated(self, sample_pdf, tmp_output, _patched_runtime):
        _, _, orch_cls, _ = _patched_runtime
        cli.main([sample_pdf, "-o", tmp_output, "--resume"])
        cfg = orch_cls.call_args[0][0]
        assert cfg.resume is True

    def test_multiple_format_flags(self, sample_pdf, tmp_output, _patched_runtime):
        _, _, orch_cls, _ = _patched_runtime
        cli.main([sample_pdf, "-o", tmp_output, "-f", "txt", "-f", "docx"])
        cfg = orch_cls.call_args[0][0]
        assert cfg.formats == ["txt", "docx"]

    def test_extract_images_and_languages(self, sample_pdf, tmp_output, _patched_runtime):
        _, _, orch_cls, _ = _patched_runtime
        cli.main([sample_pdf, "-o", tmp_output, "--extract-images", "--languages", "en,de"])
        cfg = orch_cls.call_args[0][0]
        assert cfg.extract_images is True
        assert cfg.languages == ["en", "de"]

    def test_no_force_ocr(self, sample_pdf, tmp_output, _patched_runtime):
        _, _, orch_cls, _ = _patched_runtime
        cli.main([sample_pdf, "-o", tmp_output, "--no-force-ocr"])
        cfg = orch_cls.call_args[0][0]
        assert cfg.force_ocr is False

    def test_config_file_loads_then_cli_overrides_pdfs(
        self, sample_pdf, tmp_path, _patched_runtime
    ):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({
            "pdf_paths": ["/will/be/replaced.pdf"],
            "formats": ["markdown"],
            "languages": ["fr"],
            "output_dir": str(tmp_path / "out"),
        }))

        _, _, orch_cls, _ = _patched_runtime
        rc = cli.main(["--config", str(cfg_path), sample_pdf])
        assert rc == 0

        cfg = orch_cls.call_args[0][0]
        assert cfg.pdf_paths == [sample_pdf]
        assert cfg.formats == ["markdown"]
        assert cfg.languages == ["fr"]

    def test_validation_error_returns_one(self, tmp_output, _patched_runtime, capsys):
        # Nonexistent PDF triggers OCRConfig.validate failure
        rc = cli.main(["/nope/does_not_exist.pdf", "-o", tmp_output])
        assert rc == 1
        captured = capsys.readouterr()
        assert "PDF not found" in captured.err

        _, engine, _, _ = _patched_runtime
        # Engine never loaded when validation fails
        engine.load_model.assert_not_called()

    def test_unload_called_even_on_exception(self, sample_pdf, tmp_output, _patched_runtime):
        _, engine, _, orch = _patched_runtime
        orch.run_all.side_effect = RuntimeError("boom")

        rc = cli.main([sample_pdf, "-o", tmp_output])
        assert rc == 1
        engine.unload_model.assert_called_once()

    def test_keyboard_interrupt_returns_130(self, sample_pdf, tmp_output, _patched_runtime):
        _, engine, _, orch = _patched_runtime
        orch.run_all.side_effect = KeyboardInterrupt()

        rc = cli.main([sample_pdf, "-o", tmp_output])
        assert rc == 130
        engine.unload_model.assert_called_once()


class TestCliExpandInputs:
    def test_directory_expansion_sorted(self, tmp_path, sample_pdf):
        import shutil
        d = tmp_path / "in"
        d.mkdir()
        shutil.copy(sample_pdf, d / "z.pdf")
        shutil.copy(sample_pdf, d / "a.pdf")
        shutil.copy(sample_pdf, d / "m.pdf")
        # non-PDF should be ignored
        (d / "ignore.txt").write_text("hi")

        result = cli._expand_inputs([str(d)])
        assert [p.split("/")[-1] for p in result] == ["a.pdf", "m.pdf", "z.pdf"]

    def test_file_passthrough(self, sample_pdf):
        assert cli._expand_inputs([sample_pdf]) == [sample_pdf]

    def test_mixed_inputs(self, tmp_path, sample_pdf):
        import shutil
        d = tmp_path / "in"
        d.mkdir()
        shutil.copy(sample_pdf, d / "x.pdf")
        result = cli._expand_inputs([sample_pdf, str(d)])
        assert sample_pdf in result
        assert str(d / "x.pdf") in result
