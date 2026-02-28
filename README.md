# DeepSeek OCR PRO V2

High-precision document OCR powered by the DeepSeek vision-language model. Extracts text and images from scanned PDFs with state-of-the-art accuracy.

## Features

- **CLI-first** with optional GUI - scriptable, automatable, testable
- **Multiple output formats**: TXT, TXT per page, DOCX, Markdown
- **Image extraction**: embedded images from PDF + model-detected regions
- **Checkpoint/resume**: interrupt and resume long OCR jobs
- **CPU optimized**: INT8 quantization + float32 native + thread tuning
- **Cross-platform**: Windows, macOS, Linux

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Camponotus-vagus/DeepSeek-OCR-PRO.git
cd DeepSeek-OCR-PRO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e ".[gui]"

# Download model (~6.2 GB)
deepseek-ocr --setup
```

### Usage

```bash
# Basic OCR
deepseek-ocr document.pdf

# Multiple formats with image extraction
deepseek-ocr document.pdf -f txt -f markdown -f docx --extract-images

# Resume interrupted job
deepseek-ocr large_book.pdf --resume

# Fast mode (lower accuracy, faster processing)
deepseek-ocr document.pdf -m fast

# Process all PDFs in a directory
deepseek-ocr /path/to/pdfs/ -o ./output

# Launch GUI
deepseek-ocr --gui
```

### CLI Options

```
deepseek-ocr [OPTIONS] INPUT [INPUT...]

Arguments:
  INPUT                     PDF file(s) or directory

Options:
  -o, --output DIR          Output directory (default: ./output)
  -f, --format FMT          Output format: txt, txt_pages, docx, markdown
  -m, --mode MODE           accurate or fast (default: accurate)
  --model-path DIR          Path to model directory (default: ./models)
  --quantize TYPE           none or int8 (default: int8)
  --device DEVICE           auto, cpu, cuda, mps (default: auto)
  --extract-images          Extract embedded images
  --resume                  Enable checkpoint/resume
  --gui                     Launch GUI mode
  --verbose                 Verbose logging
```

## System Requirements

- **Python**: 3.10+
- **RAM**: 16 GB minimum, 32 GB recommended
- **Disk**: ~8 GB (model weights + dependencies)
- **GPU** (optional): CUDA 11.8+ or Apple Silicon MPS

## Performance

On Intel Core i9-9880H (8 cores, 32 GB RAM) with INT8 quantization:
- Model loading: ~2 minutes
- Per-page OCR (fast mode): ~1-2 minutes
- Per-page OCR (accurate mode): ~3-5 minutes

## Project Structure

```
src/deepseek_ocr/
  cli.py            # Command-line interface
  config.py         # Configuration management
  engine/           # Model loading, OCR inference, PDF handling
  pipeline/         # Orchestrator, checkpoint, progress
  output/           # TXT, DOCX, Markdown writers
  gui/              # Optional CustomTkinter GUI
  utils/            # Device detection, logging, paths
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/deepseek_ocr --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - The vision-language model
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [PyTorch](https://pytorch.org/) - Deep learning framework
