## 2025-05-15 - [PDF Rasterization & Coordinate Parsing]
**Learning:** Directly requesting the target colorspace and disabling alpha in PyMuPDF's `get_pixmap` is more efficient than manual conversion. `json.loads` is significantly faster and safer than `eval` for parsing coordinate lists from model output.
**Action:** Always prefer `get_pixmap(..., colorspace=fitz.csRGB, alpha=False)` for RGB rendering and `json.loads` for numeric array strings.

## 2025-05-16 - [In-Memory Image Pipeline & CPU Autocast]
**Learning:** Redundant disk I/O in the OCR pipeline (saving PIL Images to temporary JPEGs for the model to re-read) added significant latency. Also, `bfloat16` autocast on CPU is emulated and much slower than native `float32`.
**Action:** Pass PIL images directly to model inference methods. Explicitly use `torch.float32` for CPU autocast to maximize native performance.
