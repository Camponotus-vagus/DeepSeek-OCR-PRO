"""Configuration management for Surya OCR PRO."""

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

    # Processing
    languages: list[str] = dataclasses.field(default_factory=lambda: ["it", "la"])
    force_ocr: bool = True  # Always OCR (needed for scanned PDFs)
    num_workers: int = 1  # Prefetch parallelism (keep low to save RAM)

    # Output
    output_dir: str = "./output"
    formats: list[str] = dataclasses.field(default_factory=lambda: ["txt"])
    extract_images: bool = False

    # Resume
    resume: bool = False

    # Logging
    verbose: bool = False
    log_file: Optional[str] = None

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
        valid_formats = {"txt", "txt_pages", "docx", "markdown"}
        for fmt in self.formats:
            if fmt not in valid_formats:
                errors.append(f"Invalid format: {fmt}. Valid: {valid_formats}")
        if self.num_workers < 1:
            errors.append("num_workers must be >= 1.")
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

        languages = [lang.strip() for lang in args.languages.split(",")] if args.languages else ["it", "la"]

        return cls(
            pdf_paths=pdf_paths,
            languages=languages,
            force_ocr=not args.no_force_ocr,
            num_workers=args.workers,
            output_dir=args.output,
            formats=args.format or ["txt"],
            extract_images=args.extract_images,
            resume=args.resume,
            verbose=args.verbose,
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
