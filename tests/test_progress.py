"""Tests for the progress reporter."""


import pytest

from surya_ocr.pipeline.progress import ProgressReporter


class _CallbackRecorder:
    def __init__(self):
        self.progress_calls = []
        self.status_calls = []

    def progress(self, current, total, eta):
        self.progress_calls.append((current, total, eta))

    def status(self, msg):
        self.status_calls.append(msg)


@pytest.fixture
def recorder():
    return _CallbackRecorder()


class TestProgressReporter:
    def test_page_start_invokes_status_callback(self, recorder):
        r = ProgressReporter(total_pages=3, pdf_name="book.pdf", status_callback=recorder.status)
        r.report_page_start(0)
        assert len(recorder.status_calls) == 1
        assert "1/3" in recorder.status_calls[0]
        assert "book.pdf" in recorder.status_calls[0]

    def test_page_done_invokes_progress_callback(self, recorder):
        r = ProgressReporter(
            total_pages=4,
            progress_callback=recorder.progress,
            status_callback=recorder.status,
        )
        r.report_page_done(page_num=0, processing_time=2.0)

        assert len(recorder.progress_calls) == 1
        current, total, eta = recorder.progress_calls[0]
        assert current == 1
        assert total == 4
        assert eta >= 0

    def test_eta_decreases_as_pages_complete(self, recorder, monkeypatch):
        # Mock time.time so wall-clock elapsed is deterministic: 1s per page.
        # Patch the `time` module on the progress module only — patching the
        # real `time.time` would also affect the logging module.
        import types
        ticks = iter([100.0, 101.0, 102.0, 103.0, 104.0])
        fake_time = types.SimpleNamespace(time=lambda: next(ticks))
        monkeypatch.setattr("surya_ocr.pipeline.progress.time", fake_time)

        r = ProgressReporter(total_pages=5, progress_callback=recorder.progress)
        for i in range(4):
            r.report_page_done(page_num=i, processing_time=1.0)

        etas = [call[2] for call in recorder.progress_calls]
        # avg = elapsed/completed = 1.0; ETA = avg * (total - completed)
        # -> [4.0, 3.0, 2.0, 1.0]
        assert etas == [4.0, 3.0, 2.0, 1.0]

    def test_eta_zero_on_last_page(self, recorder):
        r = ProgressReporter(total_pages=2, progress_callback=recorder.progress)
        r.report_page_done(page_num=0, processing_time=1.0)
        r.report_page_done(page_num=1, processing_time=1.0)
        # When completed == total_pages, remaining is 0 and eta should be 0
        assert recorder.progress_calls[-1][2] == 0.0

    def test_report_complete_sends_final_progress(self, recorder):
        r = ProgressReporter(
            total_pages=3,
            progress_callback=recorder.progress,
            status_callback=recorder.status,
        )
        r.report_complete()

        # Final progress call: (total, total, 0)
        assert recorder.progress_calls[-1] == (3, 3, 0)
        # Status mentions completion
        assert any("complete" in s.lower() for s in recorder.status_calls)

    def test_report_skipped_no_callback(self, recorder):
        # report_skipped only logs; it does NOT call back (current contract)
        r = ProgressReporter(
            total_pages=2,
            progress_callback=recorder.progress,
            status_callback=recorder.status,
        )
        r.report_skipped(0)
        assert recorder.progress_calls == []
        assert recorder.status_calls == []

    def test_report_error_calls_status_callback(self, recorder):
        r = ProgressReporter(total_pages=2, status_callback=recorder.status)
        r.report_error(1, "OCR boom")
        assert len(recorder.status_calls) == 1
        assert "page 2" in recorder.status_calls[0]
        assert "OCR boom" in recorder.status_calls[0]

    def test_no_callbacks_does_not_crash(self):
        r = ProgressReporter(total_pages=1)
        r.report_page_start(0)
        r.report_page_done(0, 1.0)
        r.report_skipped(0)
        r.report_complete()
        r.report_error(0, "x")


class TestFormatTime:
    def test_seconds_only(self):
        assert ProgressReporter._format_time(0) == "0s"
        assert ProgressReporter._format_time(45) == "45s"

    def test_minutes_and_seconds(self):
        assert ProgressReporter._format_time(60) == "1m 0s"
        assert ProgressReporter._format_time(125) == "2m 5s"

    def test_large_values(self):
        # No special 'h' formatting in the current implementation;
        # it just keeps growing in minutes.
        assert ProgressReporter._format_time(3600) == "60m 0s"

    def test_truncates_fractional(self):
        assert ProgressReporter._format_time(59.9) == "59s"
