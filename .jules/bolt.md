## 2025-05-15 - [PDF Rasterization & Coordinate Parsing]
**Learning:** Directly requesting the target colorspace and disabling alpha in PyMuPDF's `get_pixmap` is more efficient than manual conversion. `json.loads` is significantly faster and safer than `eval` for parsing coordinate lists from model output.
**Action:** Always prefer `get_pixmap(..., colorspace=fitz.csRGB, alpha=False)` for RGB rendering and `json.loads` for numeric array strings.

## 2025-05-16 - [Zero-copy Inference & CPU Autocast]
**Learning:** Passing PIL Images directly to the model's `infer` method avoids redundant disk I/O and JPEG encoding/decoding. On CPU, forced `torch.float32` in `autocast` is significantly faster than emulated `bfloat16`. `json.loads` requires `.replace("'", '"')` to safely parse LLM-generated Python-style literal strings.
**Action:** Implement zero-copy image passing for inference. Ensure CPU autocast uses `float32`. Use robust JSON parsing for model outputs.
