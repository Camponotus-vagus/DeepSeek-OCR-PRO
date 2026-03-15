## 2025-05-15 - [PDF Rasterization & Coordinate Parsing]
**Learning:** Directly requesting the target colorspace and disabling alpha in PyMuPDF's `get_pixmap` is more efficient than manual conversion. `json.loads` is significantly faster and safer than `eval` for parsing coordinate lists from model output.
**Action:** Always prefer `get_pixmap(..., colorspace=fitz.csRGB, alpha=False)` for RGB rendering and `json.loads` for numeric array strings.

## 2025-05-16 - [PIL Redundant Copies]
**Learning:** Pillow's `ImageOps.exif_transpose` and `Image.convert("RGB")` methods return a new copy of the image even if no rotation or conversion is required. This causes unnecessary memory allocations and CPU overhead, especially for high-resolution document scans.
**Action:** Always wrap these calls in conditional checks: `if img.getexif(): img = ImageOps.exif_transpose(img)` and `if img.mode != "RGB": img = img.convert("RGB")`.
