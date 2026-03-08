"""Pipeline orchestrator: coordinates PDF processing with prefetch, checkpoint, and output."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable, Optional

from ..config import OCRConfig
from ..engine.image_extractor import ImageExtractor
from ..engine.ocr_engine import OCREngine, PageResult
from ..engine.pdf_handler import PDFHandler
from ..engine.text_postprocessor import extract_grounding_regions
from ..output.writer_txt import write_txt, write_txt_per_page
from ..output.writer_docx import write_docx
from ..output.writer_markdown import write_markdown
from .checkpoint import CheckpointManager
from .progress import ProgressReporter

log = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the full OCR pipeline for one or more PDFs.

    Features:
    - Page prefetching (extracts next pages while model processes current)
    - Checkpoint/resume support
    - Multiple output format generation
    - Image extraction (embedded + model-detected regions)
    """

    def __init__(
        self,
        config: OCRConfig,
        engine: OCREngine,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ):
        self.config = config
        self.engine = engine
        self.pdf_handler = PDFHandler()
        self.image_extractor = ImageExtractor()
        self._progress_cb = progress_callback
        self._status_cb = status_callback
        self._cancel_check = cancel_check

    def run_all(self) -> None:
        """Process all PDFs in the config."""
        for i, pdf_path in enumerate(self.config.pdf_paths):
            log.info(f"Processing PDF {i + 1}/{len(self.config.pdf_paths)}: {pdf_path}")
            if self._status_cb:
                self._status_cb(f"Processing: {Path(pdf_path).name}")
            self.run_single(pdf_path)

    def run_single(self, pdf_path: str) -> None:
        """Process a single PDF file through the full pipeline."""
        pdf_name = Path(pdf_path).name
        total_pages = self.pdf_handler.get_page_count(pdf_path)

        if total_pages == 0:
            log.error(f"Could not read PDF or PDF is empty: {pdf_path}")
            return

        log.info(f"PDF '{pdf_name}': {total_pages} pages")

        # Setup output directory
        output_subdir = Path(self.config.output_dir) / Path(pdf_path).stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint
        checkpoint = CheckpointManager(pdf_path, self.config.output_dir)
        completed_pages: set[int] = set()

        if self.config.resume and checkpoint.is_valid(total_pages):
            completed_pages = checkpoint.get_completed_pages()
            log.info(f"Resuming: {len(completed_pages)}/{total_pages} pages already done")
        else:
            checkpoint.init(total_pages, str(self.config.mode))

        # Setup progress
        progress = ProgressReporter(
            total_pages=total_pages,
            pdf_name=pdf_name,
            progress_callback=self._progress_cb,
            status_callback=self._status_cb,
        )

        # Collect results (load cached + process remaining)
        results: dict[int, PageResult] = {}

        for page_num in completed_pages:
            try:
                results[page_num] = checkpoint.load_page(page_num)
                progress.report_skipped(page_num)
            except Exception as e:
                log.warning(f"Failed to load checkpoint for page {page_num}: {e}")
                completed_pages.discard(page_num)

        # Process remaining pages with prefetch
        pages_to_process = [p for p in range(total_pages) if p not in completed_pages]

        if pages_to_process:
            self._process_pages_with_prefetch(
                pdf_path, pages_to_process, total_pages,
                results, checkpoint, progress, str(output_subdir),
            )

        # Check cancellation
        if self._is_cancelled():
            log.info("Processing cancelled by user")
            return

        progress.report_complete()

        # Generate final outputs
        ordered_results = [results.get(i) for i in range(total_pages)]
        ordered_results = [r for r in ordered_results if r is not None]

        self._write_outputs(ordered_results, pdf_path, str(output_subdir))

        # Cleanup checkpoint
        checkpoint.cleanup()
        log.info(f"Completed: {pdf_name}")

    def _process_pages_with_prefetch(
        self,
        pdf_path: str,
        pages_to_process: list[int],
        total_pages: int,
        results: dict[int, PageResult],
        checkpoint: CheckpointManager,
        progress: ProgressReporter,
        output_dir: str,
    ) -> None:
        """Process pages with background prefetching of next pages."""
        prefetch_count = min(self.config.num_workers, len(pages_to_process))

        with ThreadPoolExecutor(max_workers=prefetch_count) as pool:
            # Submit initial prefetch batch
            futures: dict[int, Future] = {}
            for page_num in pages_to_process[:prefetch_count]:
                futures[page_num] = pool.submit(
                    self.pdf_handler.extract_page_image, pdf_path, page_num
                )

            submitted_idx = prefetch_count

            for page_num in pages_to_process:
                if self._is_cancelled():
                    break

                progress.report_page_start(page_num)

                # Get prefetched image (or extract now if not prefetched)
                if page_num in futures:
                    try:
                        image = futures.pop(page_num).result(timeout=120)
                    except Exception as e:
                        log.error(f"Failed to extract page {page_num}: {e}")
                        result = PageResult(page_num, "", 0, error=str(e))
                        results[page_num] = result
                        checkpoint.save_page(result)
                        progress.report_error(page_num, str(e))
                        continue
                else:
                    try:
                        image = self.pdf_handler.extract_page_image(pdf_path, page_num)
                    except Exception as e:
                        log.error(f"Failed to extract page {page_num}: {e}")
                        result = PageResult(page_num, "", 0, error=str(e))
                        results[page_num] = result
                        checkpoint.save_page(result)
                        progress.report_error(page_num, str(e))
                        continue

                # Submit next prefetch
                if submitted_idx < len(pages_to_process):
                    next_page = pages_to_process[submitted_idx]
                    futures[next_page] = pool.submit(
                        self.pdf_handler.extract_page_image, pdf_path, next_page
                    )
                    submitted_idx += 1

                # Run OCR
                result = self.engine.process_page(image, page_num)
                results[page_num] = result
                checkpoint.save_page(result)

                # Extract images if requested
                if self.config.extract_images and result.raw_text:
                    self._extract_page_images(
                        pdf_path, page_num, image, result.raw_text, output_dir
                    )

                if result.error:
                    progress.report_error(page_num, result.error)
                else:
                    progress.report_page_done(page_num, result.processing_time)

                # Free page image memory between pages
                del image
                import gc
                gc.collect()

    def _extract_page_images(
        self, pdf_path: str, page_num: int, page_image, raw_text: str, output_dir: str
    ) -> None:
        """Extract both embedded images and model-detected regions."""
        try:
            # Embedded images from PDF
            self.image_extractor.extract_embedded_images(pdf_path, page_num, output_dir)

            # Model-detected image regions
            regions = extract_grounding_regions(raw_text)
            if regions:
                self.image_extractor.extract_grounding_regions(
                    page_image, regions, page_num, output_dir
                )
        except Exception as e:
            log.warning(f"Image extraction failed for page {page_num}: {e}")

    def _write_outputs(
        self, results: list[PageResult], pdf_path: str, output_dir: str
    ) -> None:
        """Generate all requested output formats."""
        if not results:
            log.warning("No results to write")
            return

        pdf_stem = Path(pdf_path).stem

        for fmt in self.config.formats:
            try:
                if fmt == "txt":
                    write_txt(results, output_dir, pdf_stem)
                elif fmt == "txt_pages":
                    write_txt_per_page(results, output_dir, pdf_stem)
                elif fmt == "docx":
                    write_docx(results, output_dir, pdf_stem)
                elif fmt == "markdown":
                    write_markdown(results, output_dir, pdf_stem)
                log.info(f"Output written: {fmt}")
            except Exception as e:
                log.error(f"Failed to write {fmt} output: {e}")

    def _is_cancelled(self) -> bool:
        return self._cancel_check is not None and self._cancel_check()
