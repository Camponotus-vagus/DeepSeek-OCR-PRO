"""Configuration management for DeepSeek OCR PRO."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class OCRConfig:
    """Central configuration for an OCR job."""

    # Input
    pdf_paths: list[str] = dataclasses.field(default_factory=list)

    # Model
    model_path: str = "./models"
    quantize: str = "none"  # "none" or "int8" (int8 hurts OCR quality on CPU)
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Processing
    # NOTE: For scanned documents (B&W 1-bit, ~200 DPI), always use "accurate" mode.
    # "fast" mode (640px) downscales too aggressively for dense text pages.
    mode: str = "accurate"  # "accurate" (1024px, crop) or "fast" (640px, no crop)
    num_workers: int = 1  # Preprocessing parallelism (keep low to save RAM)

    # Output
    output_dir: str = "./output"
    formats: list[str] = dataclasses.field(default_factory=lambda: ["txt"])
    extract_images: bool = False

    # Generation
    max_new_tokens: int = 4096  # Cap model generation length (lower = faster per page)

    # Resume
    resume: bool = False

    # Prompt
    prompt_mode: str = "layout"  # "layout" (markdown) or "freeocr" (plain text)

    # Logging
    verbose: bool = False
    log_file: Optional[str] = None

    @property
    def base_size(self) -> int:
        return 640 if self.mode == "fast" else 1024

    @property
    def crop_mode(self) -> bool:
        return self.mode != "fast"

    @property
    def prompt(self) -> str:
        if self.prompt_mode == "layout":
            return "<image>\n<|grounding|>Convert the document to markdown. "
        return "<image>\nFree OCR. "

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors = []
        if not self.pdf_paths:
            errors.append("No PDF files specified.")
        for p in self.pdf_paths:
            if not Path(p).exists():
                errors.append(f"PDF not found: {p}")
            elif not p.lower().endswith(".pdf"):
                errors.append(f"Not a PDF file: {p}")
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            errors.append(f"Model directory not found: {self.model_path}")
        elif not (model_dir / "config.json").exists():
            errors.append(f"Model config.json not found in: {self.model_path}")
        if self.mode not in ("accurate", "fast"):
            errors.append(f"Invalid mode: {self.mode}. Use 'accurate' or 'fast'.")
        if self.quantize not in ("none", "int8"):
            errors.append(f"Invalid quantize: {self.quantize}. Use 'none' or 'int8'.")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            errors.append(f"Invalid device: {self.device}. Use 'auto', 'cpu', 'cuda', or 'mps'.")
        valid_formats = {"txt", "txt_pages", "docx", "markdown"}
        for fmt in self.formats:
            if fmt not in valid_formats:
                errors.append(f"Invalid format: {fmt}. Valid: {valid_formats}")
        if self.prompt_mode not in ("layout", "freeocr"):
            errors.append(f"Invalid prompt_mode: {self.prompt_mode}.")
        if self.num_workers < 1:
            errors.append("num_workers must be >= 1.")
        if self.max_new_tokens < 128:
            errors.append("max_new_tokens must be >= 128.")
        return errors

    @classmethod
    def from_file(cls, path: str) -> OCRConfig:
        """Load config from a JSON or YAML file."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})

    @classmethod
    def from_args(cls, args) -> OCRConfig:
        """Build config from parsed argparse namespace."""
        pdf_paths = []
        for inp in args.inputs:
            p = Path(inp)
            if p.is_dir():
                pdf_paths.extend(str(f) for f in sorted(p.glob("*.pdf")))
            else:
                pdf_paths.append(str(p))

        return cls(
            pdf_paths=pdf_paths,
            model_path=args.model_path,
            quantize=args.quantize,
            device=args.device,
            mode=args.mode,
            num_workers=args.workers,
            output_dir=args.output,
            formats=args.format or ["txt"],
            extract_images=args.extract_images,
            max_new_tokens=getattr(args, "max_tokens", 4096),
            resume=args.resume,
            prompt_mode=args.prompt,
            verbose=args.verbose,
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
