"""Tests for logging setup."""

import logging

import pytest

from surya_ocr.utils.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Snapshot root logger state and restore after each test."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = root.handlers[:]
    yield
    root.setLevel(saved_level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)


class TestSetupLogging:
    def test_default_adds_console_handler(self):
        setup_logging()
        root = logging.getLogger()
        # Exactly one handler (the console) when no log_file is given
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_verbose_sets_debug_on_console(self):
        setup_logging(verbose=True)
        root = logging.getLogger()
        console = root.handlers[0]
        assert console.level == logging.DEBUG

    def test_non_verbose_sets_info_on_console(self):
        setup_logging(verbose=False)
        root = logging.getLogger()
        console = root.handlers[0]
        assert console.level == logging.INFO

    def test_file_handler_added_when_log_file_given(self, tmp_path):
        log_file = tmp_path / "logs" / "app.log"
        setup_logging(verbose=False, log_file=str(log_file))

        root = logging.getLogger()
        assert len(root.handlers) == 2
        # Parent dir was created
        assert log_file.parent.exists()

        # Writing through the logger reaches the file
        logging.getLogger("test_logger").error("hello")
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
        assert "hello" in log_file.read_text(encoding="utf-8")

    def test_noisy_libraries_silenced(self):
        setup_logging()
        for name in ("transformers", "torch", "PIL"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_replaces_existing_handlers(self):
        root = logging.getLogger()
        sentinel = logging.NullHandler()
        root.addHandler(sentinel)

        setup_logging()
        assert sentinel not in root.handlers
