"""Core OCR engine wrapping the DeepSeek model."""

from __future__ import annotations

import dataclasses
import logging
import tempfile
import time

from PIL import Image

from ..config import OCRConfig

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PageResult:
    """Result of OCR processing for a single page."""

    page_number: int
    raw_text: str
    processing_time: float
    error: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PageResult:
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})


class OCREngine:
    """Stateful OCR engine that holds the loaded model.

    Usage:
        engine = OCREngine(config)
        engine.load_model()
        result = engine.process_page(image, page_num=0)
        engine.unload_model()
    """

    def __init__(self, config: OCRConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the model into memory with configured optimizations."""
        if self._loaded:
            log.info("Model already loaded, skipping")
            return

        from ..utils.device import configure_cpu_threads, detect_device
        from .model_loader import load_model

        self._device = detect_device(self.config.device)

        if self._device == "cpu":
            configure_cpu_threads()

        log.info(f"Loading model on {self._device} with quantize={self.config.quantize}")
        self._model, self._tokenizer = load_model(
            model_path=self.config.model_path,
            device=self._device,
            quantize=self.config.quantize,
        )
        self._loaded = True
        log.info("Model ready for inference")

    def process_page(self, image: Image.Image, page_num: int = 0) -> PageResult:
        """Run OCR inference on a single page image.

        Args:
            image: PIL Image of the page (any mode, will be converted to RGB).
            page_num: Page number for metadata.

        Returns:
            PageResult with extracted text and timing info.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.time()

        try:
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Model now supports direct PIL Image passing
            with tempfile.TemporaryDirectory() as tmpdir:
                text = self._run_inference(image, tmpdir)

            elapsed = time.time() - start
            log.info(f"Page {page_num + 1} processed in {elapsed:.1f}s ({len(text)} chars)")

            return PageResult(
                page_number=page_num,
                raw_text=text,
                processing_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            error_msg = f"OCR failed for page {page_num + 1}: {e}"
            log.error(error_msg)
            return PageResult(
                page_number=page_num,
                raw_text="",
                processing_time=elapsed,
                error=error_msg,
            )

    def _run_inference(self, image: Image.Image | str, output_dir: str) -> str:
        """Call the model's infer() method and return decoded text."""
        try:
            result = self._model.infer(
                self._tokenizer,
                prompt=self.config.prompt,
                image_file=image,
                output_path=output_dir,
                base_size=self.config.base_size,
                image_size=self.config.base_size,
                crop_mode=self.config.crop_mode,
                eval_mode=True,  # CRITICAL: enables text return instead of streaming
            )
        except Exception:
            raise

        if result is None:
            return ""

        # Result can be a string or tuple
        text = str(result[0]) if isinstance(result, tuple) else str(result)

        # Clean up EOS tokens that might slip through
        eos_markers = ["<\u2581end\u2581of\u2581sentence\u2581>", "<|end▁of▁sentence|>"]
        for marker in eos_markers:
            text = text.replace(marker, "")

        return text.strip()

    def unload_model(self) -> None:
        """Free model memory."""
        if not self._loaded:
            return

        from .model_loader import unload_model
        unload_model(self._model, self._tokenizer)
        self._model = None
        self._tokenizer = None
        self._loaded = False
