"""Tests for the pipeline orchestrator (engine and writers mocked)."""

from pathlib import Path
from unittest.mock import MagicMock, patch


from surya_ocr.config import OCRConfig
from surya_ocr.engine.ocr_engine import OCREngine, PageResult
from surya_ocr.pipeline.checkpoint import CheckpointManager
from surya_ocr.pipeline.orchestrator import Orchestrator


def _mock_engine(canned_text="canned page text"):
    """Build a MagicMock(spec=OCREngine) whose process_pdf_by_page returns canned results."""
    engine = MagicMock(spec=OCREngine)
    engine.process_pdf_by_page.side_effect = lambda pdf, page_num: PageResult(
        page_number=page_num,
        raw_text=f"{canned_text} {page_num}",
        processing_time=0.5,
    )
    return engine


def _make_config(pdf_path, output_dir, **overrides):
    kwargs = dict(
        pdf_paths=[pdf_path],
        output_dir=output_dir,
        formats=["txt"],
    )
    kwargs.update(overrides)
    return OCRConfig(**kwargs)


# ---------- happy path ----------

class TestRunSingleHappyPath:
    def test_processes_every_page_and_writes_output(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output, formats=["txt"])
        engine = _mock_engine()

        Orchestrator(cfg, engine).run_single(multi_page_pdf)

        # process_pdf_by_page called once per page (5 pages in fixture)
        assert engine.process_pdf_by_page.call_count == 5
        # Output file present
        pdf_stem = Path(multi_page_pdf).stem
        output_file = Path(tmp_output) / pdf_stem / f"{pdf_stem}.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        for i in range(5):
            assert f"canned page text {i}" in content

    def test_cleans_checkpoint_after_success(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output)
        Orchestrator(cfg, _mock_engine()).run_single(multi_page_pdf)

        checkpoint_dir = Path(tmp_output) / Path(multi_page_pdf).stem / ".checkpoint"
        assert not checkpoint_dir.exists()

    def test_writes_multiple_formats(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output, formats=["txt", "markdown"])
        Orchestrator(cfg, _mock_engine()).run_single(multi_page_pdf)

        stem = Path(multi_page_pdf).stem
        assert (Path(tmp_output) / stem / f"{stem}.txt").exists()
        assert (Path(tmp_output) / stem / f"{stem}.md").exists()


# ---------- run_all ----------

class TestRunAll:
    def test_iterates_all_pdfs(self, sample_pdf, multi_page_pdf, tmp_output):
        cfg = OCRConfig(
            pdf_paths=[sample_pdf, multi_page_pdf],
            output_dir=tmp_output,
            formats=["txt"],
        )
        engine = _mock_engine()
        Orchestrator(cfg, engine).run_all()

        # 1 + 5 = 6 page calls total
        assert engine.process_pdf_by_page.call_count == 6

    def test_status_callback_called_per_pdf(self, sample_pdf, multi_page_pdf, tmp_output):
        cfg = OCRConfig(
            pdf_paths=[sample_pdf, multi_page_pdf],
            output_dir=tmp_output,
        )
        status_msgs = []
        Orchestrator(
            cfg,
            _mock_engine(),
            status_callback=status_msgs.append,
        ).run_all()

        # At least one status update naming each PDF
        joined = "\n".join(status_msgs)
        assert Path(sample_pdf).name in joined
        assert Path(multi_page_pdf).name in joined


# ---------- resume ----------

class TestResume:
    def test_skips_already_completed_pages(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output, resume=True)

        # Pre-populate checkpoint with pages 0, 1, 2 (total = 5 pages)
        cm = CheckpointManager(multi_page_pdf, tmp_output)
        cm.init(5, "marker-pdf")
        for i in range(3):
            cm.save_page(PageResult(page_number=i, raw_text=f"cached {i}", processing_time=0.0))

        engine = _mock_engine()
        Orchestrator(cfg, engine).run_single(multi_page_pdf)

        # Only pages 3 and 4 should hit the engine
        assert engine.process_pdf_by_page.call_count == 2
        called_pages = {c.args[1] for c in engine.process_pdf_by_page.call_args_list}
        assert called_pages == {3, 4}

        # Output contains the cached text for the resumed pages
        stem = Path(multi_page_pdf).stem
        output = (Path(tmp_output) / stem / f"{stem}.txt").read_text(encoding="utf-8")
        assert "cached 0" in output
        assert "cached 2" in output

    def test_corrupt_checkpoint_falls_through_to_full_run(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output, resume=True)

        # Init with a stale page count → is_valid() returns False
        cm = CheckpointManager(multi_page_pdf, tmp_output)
        cm.init(99, "marker-pdf")  # mismatched total_pages
        for i in range(2):
            cm.save_page(PageResult(page_number=i, raw_text=f"stale {i}", processing_time=0.0))

        engine = _mock_engine()
        Orchestrator(cfg, engine).run_single(multi_page_pdf)

        # All 5 pages re-processed
        assert engine.process_pdf_by_page.call_count == 5


# ---------- cancellation ----------

class TestCancellation:
    def test_cancel_check_stops_processing(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output)

        calls = {"n": 0}

        def cancel_after_two():
            calls["n"] += 1
            return calls["n"] > 2  # False, False, True, ...

        engine = _mock_engine()
        Orchestrator(cfg, engine, cancel_check=cancel_after_two).run_single(multi_page_pdf)

        # Loop exits before all 5 pages processed
        assert engine.process_pdf_by_page.call_count < 5
        # No final output written when cancelled
        stem = Path(multi_page_pdf).stem
        assert not (Path(tmp_output) / stem / f"{stem}.txt").exists()


# ---------- error containment ----------

class TestErrorContainment:
    def test_engine_error_does_not_abort_loop(self, multi_page_pdf, tmp_output):
        cfg = _make_config(multi_page_pdf, tmp_output)

        def maybe_fail(pdf, page_num):
            if page_num == 2:
                return PageResult(
                    page_number=page_num, raw_text="", processing_time=0.1, error="page 3 boom"
                )
            return PageResult(page_number=page_num, raw_text=f"text {page_num}", processing_time=0.5)

        engine = MagicMock(spec=OCREngine)
        engine.process_pdf_by_page.side_effect = maybe_fail

        Orchestrator(cfg, engine).run_single(multi_page_pdf)

        # All 5 pages attempted despite one error
        assert engine.process_pdf_by_page.call_count == 5
        # Error placeholder appears in output
        stem = Path(multi_page_pdf).stem
        content = (Path(tmp_output) / stem / f"{stem}.txt").read_text(encoding="utf-8")
        assert "page 3 boom" in content
        assert "text 0" in content
        assert "text 4" in content

    def test_image_extraction_skipped_for_errored_pages(self, sample_pdf, tmp_output):
        cfg = _make_config(sample_pdf, tmp_output, extract_images=True)

        engine = MagicMock(spec=OCREngine)
        engine.process_pdf_by_page.return_value = PageResult(
            page_number=0, raw_text="", processing_time=0.1, error="bad"
        )

        with patch("surya_ocr.engine.image_extractor.ImageExtractor") as mock_extractor_cls:
            Orchestrator(cfg, engine).run_single(sample_pdf)

        mock_extractor_cls.assert_not_called()


# ---------- image extraction ----------

class TestImageExtraction:
    def test_extract_images_invokes_extractor(self, sample_pdf, tmp_output):
        cfg = _make_config(sample_pdf, tmp_output, extract_images=True)
        engine = _mock_engine()

        with patch("surya_ocr.engine.image_extractor.ImageExtractor") as mock_cls:
            instance = mock_cls.return_value
            Orchestrator(cfg, engine).run_single(sample_pdf)

        instance.extract_embedded_images.assert_called()

    def test_extract_images_failure_logged_not_raised(self, sample_pdf, tmp_output):
        cfg = _make_config(sample_pdf, tmp_output, extract_images=True)
        engine = _mock_engine()

        with patch("surya_ocr.engine.image_extractor.ImageExtractor") as mock_cls:
            mock_cls.return_value.extract_embedded_images.side_effect = RuntimeError("disk full")
            # Should NOT raise — failure is logged as a warning
            Orchestrator(cfg, engine).run_single(sample_pdf)


# ---------- format dispatch ----------

class TestFormatDispatch:
    def test_dispatches_each_format(self, sample_pdf, tmp_output):
        cfg = _make_config(
            sample_pdf, tmp_output, formats=["txt", "txt_pages", "markdown", "docx"]
        )
        engine = _mock_engine()

        with patch("surya_ocr.pipeline.orchestrator.write_txt") as txt, \
             patch("surya_ocr.pipeline.orchestrator.write_txt_per_page") as txt_pages, \
             patch("surya_ocr.pipeline.orchestrator.write_markdown") as md, \
             patch("surya_ocr.pipeline.orchestrator.write_docx") as docx:
            Orchestrator(cfg, engine).run_single(sample_pdf)

        txt.assert_called_once()
        txt_pages.assert_called_once()
        md.assert_called_once()
        docx.assert_called_once()

    def test_writer_exception_does_not_abort_other_formats(self, sample_pdf, tmp_output):
        cfg = _make_config(sample_pdf, tmp_output, formats=["txt", "markdown"])
        engine = _mock_engine()

        with patch("surya_ocr.pipeline.orchestrator.write_txt", side_effect=RuntimeError("io")), \
             patch("surya_ocr.pipeline.orchestrator.write_markdown") as md:
            # Should not raise
            Orchestrator(cfg, engine).run_single(sample_pdf)

        md.assert_called_once()

    def test_unknown_format_silently_ignored(self, sample_pdf, tmp_output):
        cfg = _make_config(sample_pdf, tmp_output, formats=["txt", "weird"])
        engine = _mock_engine()
        # Should not raise even though "weird" is not a known format
        Orchestrator(cfg, engine).run_single(sample_pdf)

        stem = Path(sample_pdf).stem
        assert (Path(tmp_output) / stem / f"{stem}.txt").exists()


# ---------- edge cases ----------

class TestEdgeCases:
    def test_zero_page_pdf_returns_early(self, tmp_output):
        cfg = _make_config("/nonexistent.pdf", tmp_output)
        engine = _mock_engine()

        with patch.object(
            Orchestrator, "_extract_page_images"  # not called
        ), patch("surya_ocr.pipeline.orchestrator.PDFHandler") as mock_handler_cls:
            mock_handler_cls.return_value.get_page_count.return_value = 0
            Orchestrator(cfg, engine).run_single("/nonexistent.pdf")

        engine.process_pdf_by_page.assert_not_called()
