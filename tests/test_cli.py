"""Tests for CLI argument parsing."""

import pytest

from deepseek_ocr.cli import build_parser, main


class TestCLIParser:
    def setup_method(self):
        self.parser = build_parser()

    def test_basic_args(self):
        args = self.parser.parse_args(["input.pdf"])
        assert args.inputs == ["input.pdf"]
        assert args.output == "./output"
        assert args.mode == "accurate"
        assert args.quantize == "int8"
        assert args.device == "auto"

    def test_multiple_inputs(self):
        args = self.parser.parse_args(["a.pdf", "b.pdf", "c.pdf"])
        assert args.inputs == ["a.pdf", "b.pdf", "c.pdf"]

    def test_output_dir(self):
        args = self.parser.parse_args(["in.pdf", "-o", "/tmp/out"])
        assert args.output == "/tmp/out"

    def test_formats(self):
        args = self.parser.parse_args(["in.pdf", "-f", "txt", "-f", "markdown", "-f", "docx"])
        assert args.format == ["txt", "markdown", "docx"]

    def test_fast_mode(self):
        args = self.parser.parse_args(["in.pdf", "-m", "fast"])
        assert args.mode == "fast"

    def test_quantize_none(self):
        args = self.parser.parse_args(["in.pdf", "--quantize", "none"])
        assert args.quantize == "none"

    def test_device_cpu(self):
        args = self.parser.parse_args(["in.pdf", "--device", "cpu"])
        assert args.device == "cpu"

    def test_extract_images(self):
        args = self.parser.parse_args(["in.pdf", "--extract-images"])
        assert args.extract_images is True

    def test_resume(self):
        args = self.parser.parse_args(["in.pdf", "--resume"])
        assert args.resume is True

    def test_workers(self):
        args = self.parser.parse_args(["in.pdf", "--workers", "4"])
        assert args.workers == 4

    def test_prompt_freeocr(self):
        args = self.parser.parse_args(["in.pdf", "--prompt", "freeocr"])
        assert args.prompt == "freeocr"

    def test_gui_flag(self):
        args = self.parser.parse_args(["--gui"])
        assert args.gui is True

    def test_setup_flag(self):
        args = self.parser.parse_args(["--setup"])
        assert args.setup is True

    def test_verbose(self):
        args = self.parser.parse_args(["in.pdf", "--verbose"])
        assert args.verbose is True

    def test_config_file(self):
        args = self.parser.parse_args(["--config", "my_config.yaml", "in.pdf"])
        assert args.config == "my_config.yaml"

    def test_no_inputs_launches_gui(self, monkeypatch):
        """CLI should launch GUI when no arguments are provided (double-click)."""
        launched = []
        monkeypatch.setattr(
            "deepseek_ocr.cli._launch_gui",
            lambda: launched.append(True) or 0,
        )
        result = main([])
        assert result == 0
        assert launched, "GUI should be launched when no arguments are given"


class TestCLIHelp:
    def test_help_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_version_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0
