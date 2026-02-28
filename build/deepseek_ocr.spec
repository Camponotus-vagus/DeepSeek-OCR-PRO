# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DeepSeek OCR PRO V2.

Build with:
    pyinstaller build/deepseek_ocr.spec

The resulting executable does NOT include model weights (6.2 GB).
Users download the model on first run via: deepseek-ocr --setup
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Project root (one level up from build/)
project_root = Path(SPECPATH).parent

# Model config files to bundle (NOT the weights)
model_dir = project_root / "models"
model_data_files = []
model_config_files = [
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "modeling_deepseekocr.py",
    "modeling_deepseekv2.py",
    "deepencoder.py",
    "conversation.py",
    "configuration_deepseek_v2.py",
]

for fname in model_config_files:
    fpath = model_dir / fname
    if fpath.exists():
        model_data_files.append((str(fpath), "models"))

a = Analysis(
    [str(project_root / "src" / "deepseek_ocr" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=model_data_files,
    hiddenimports=[
        "customtkinter",
        "torch",
        "transformers",
        "safetensors",
        "einops",
        "easydict",
        "addict",
        "deepseek_ocr",
        "deepseek_ocr.cli",
        "deepseek_ocr.config",
        "deepseek_ocr.engine",
        "deepseek_ocr.engine.model_loader",
        "deepseek_ocr.engine.ocr_engine",
        "deepseek_ocr.engine.pdf_handler",
        "deepseek_ocr.engine.text_postprocessor",
        "deepseek_ocr.engine.image_extractor",
        "deepseek_ocr.pipeline",
        "deepseek_ocr.pipeline.orchestrator",
        "deepseek_ocr.pipeline.checkpoint",
        "deepseek_ocr.pipeline.progress",
        "deepseek_ocr.output",
        "deepseek_ocr.output.writer_txt",
        "deepseek_ocr.output.writer_docx",
        "deepseek_ocr.output.writer_markdown",
        "deepseek_ocr.gui",
        "deepseek_ocr.gui.app",
        "deepseek_ocr.gui.pdf_preview",
        "deepseek_ocr.utils",
        "deepseek_ocr.utils.device",
        "deepseek_ocr.utils.logging_setup",
        "deepseek_ocr.utils.paths",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "notebook",
        "jupyter",
        "scipy",
        "pandas",
        "sklearn",
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

# Collect customtkinter data files (themes, etc.)
try:
    import customtkinter
    ctk_path = Path(customtkinter.__file__).parent
    for root, dirs, files in os.walk(ctk_path):
        for f in files:
            full = Path(root) / f
            rel = full.relative_to(ctk_path.parent)
            a.datas.append((str(full), str(rel.parent), "DATA"))
except ImportError:
    pass

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="deepseek-ocr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
