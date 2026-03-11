## 2025-05-15 - [PDF Rasterization & Coordinate Parsing]
**Learning:** Directly requesting the target colorspace and disabling alpha in PyMuPDF's `get_pixmap` is more efficient than manual conversion. `json.loads` is significantly faster and safer than `eval` for parsing coordinate lists from model output.
**Action:** Always prefer `get_pixmap(..., colorspace=fitz.csRGB, alpha=False)` for RGB rendering and `json.loads` for numeric array strings.
