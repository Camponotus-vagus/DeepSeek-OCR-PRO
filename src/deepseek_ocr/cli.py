"""Command-line interface for DeepSeek OCR PRO."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek-ocr",
        description="DeepSeek OCR PRO - High-precision document OCR powered by DeepSeek vision-language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  deepseek-ocr book.pdf
  deepseek-ocr book.pdf -f txt -f markdown --extract-images --resume
  deepseek-ocr /path/to/pdfs/ -o ./output --quantize int8
  deepseek-ocr --gui
  deepseek-ocr --setup
        """,
    )

    parser.add_argument(
        "inputs", nargs="*", metavar="INPUT",
        help="PDF file(s) or directory containing PDFs",
    )
    parser.add_argument(
        "-o", "--output", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "-f", "--format", action="append", default=None,
        choices=["txt", "txt_pages", "docx", "markdown"],
        help="Output format (can be specified multiple times, default: txt)",
    )
    parser.add_argument(
        "-m", "--mode", default="accurate", choices=["accurate", "fast"],
        help="OCR mode: accurate (1024px, crop) or fast (640px, default: accurate)",
    )
    parser.add_argument(
        "--model-path", default="./models",
        help="Path to model directory (default: ./models)",
    )
    parser.add_argument(
        "--quantize", default="int8", choices=["none", "int8"],
        help="Quantization: none or int8 (default: int8)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
        help="Device: auto, cpu, cuda, mps (default: auto)",
    )
    parser.add_argument(
        "--extract-images", action="store_true",
        help="Extract embedded images and model-detected regions",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Enable checkpoint/resume for interrupted jobs",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of prefetch workers (default: 2)",
    )
    parser.add_argument(
        "--prompt", default="layout", choices=["layout", "freeocr"],
        help="Prompt mode: layout (markdown output) or freeocr (plain text, default: layout)",
    )
    parser.add_argument(
        "--config", metavar="FILE",
        help="Load config from YAML/JSON file",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch GUI mode (default when no arguments are given)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Download model from HuggingFace",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # GUI mode — also launch GUI when no arguments are provided at all
    # (e.g. user double-clicks the .exe)
    if args.gui or (not args.inputs and not args.setup):
        return _launch_gui()

    # Setup/download mode
    if args.setup:
        return _run_setup(args)

    # Build config
    from .config import OCRConfig
    from .utils.logging_setup import setup_logging
    from .utils.paths import resolve_model_path

    setup_logging(verbose=args.verbose)

    if args.config:
        config = OCRConfig.from_file(args.config)
        # Override with CLI args
        if args.inputs:
            config.pdf_paths = _expand_inputs(args.inputs)
    else:
        # Resolve model path
        args.model_path = str(resolve_model_path(args.model_path))
        config = OCRConfig.from_args(args)

    # Validate
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    # Run OCR
    return _run_ocr(config)


def _run_ocr(config) -> int:
    """Execute the OCR pipeline."""
    import logging
    from .engine.ocr_engine import OCREngine
    from .pipeline.orchestrator import Orchestrator

    log = logging.getLogger(__name__)

    engine = OCREngine(config)

    try:
        log.info("Loading model...")
        engine.load_model()

        orchestrator = Orchestrator(config, engine)
        orchestrator.run_all()

        log.info("All PDFs processed successfully")
        return 0

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 130

    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        engine.unload_model()


def _launch_gui() -> int:
    """Launch the GUI application."""
    try:
        from .gui.app import launch_gui
        launch_gui()
        return 0
    except ImportError:
        print("GUI requires customtkinter. Install with: pip install customtkinter", file=sys.stderr)
        return 1


def _run_setup(args) -> int:
    """Download the model from HuggingFace."""
    from .utils.logging_setup import setup_logging
    setup_logging(verbose=args.verbose)

    try:
        from huggingface_hub import snapshot_download
        import logging

        log = logging.getLogger(__name__)
        model_path = args.model_path

        log.info(f"Downloading DeepSeek-OCR model to {model_path}")
        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-OCR",
            local_dir=model_path,
        )
        log.info("Model download complete")
        return 0

    except Exception as e:
        print(f"Setup failed: {e}", file=sys.stderr)
        return 1


def _expand_inputs(inputs: list[str]) -> list[str]:
    """Expand directory inputs to individual PDF paths."""
    from pathlib import Path
    pdf_paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            pdf_paths.extend(str(f) for f in sorted(p.glob("*.pdf")))
        else:
            pdf_paths.append(str(p))
    return pdf_paths
