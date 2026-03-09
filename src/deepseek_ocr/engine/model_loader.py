"""Model loading with optimization for CPU inference."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

log = logging.getLogger(__name__)


def load_model(model_path: str, device: str, quantize: str = "int8"):
    """Load the DeepSeek-OCR model and tokenizer with optimizations.

    Args:
        model_path: Path to the model directory containing config.json and weights.
        device: Target device ("cpu", "cuda", "mps").
        quantize: Quantization strategy ("none" or "int8").

    Returns:
        Tuple of (model, tokenizer).
    """
    # Suppress noisy warnings during loading
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = str(Path(model_path).resolve())
    log.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Choose dtype based on device
    if device == "mps":
        load_dtype = torch.float16
    elif device == "cpu":
        # float32 is native on Intel CPUs; bfloat16 is emulated and slow
        load_dtype = torch.float32
    else:
        # CUDA: use bfloat16 (native on modern GPUs)
        load_dtype = torch.bfloat16

    log.info(f"Loading model with dtype={load_dtype} for device={device}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,  # Avoid doubling peak memory during loading
    )
    model = model.to(torch.device(device)).eval()

    # Apply INT8 dynamic quantization on CPU
    if quantize == "int8" and device == "cpu":
        log.info("Applying INT8 dynamic quantization to language model linear layers")
        model = _apply_int8_quantization(model)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model loaded: {param_count:.0f}M parameters on {device}")

    return model, tokenizer


def _apply_int8_quantization(model):
    """Apply PyTorch dynamic INT8 quantization to linear layers.

    Quantizes one transformer layer at a time to avoid doubling the
    model's memory footprint (critical on systems with limited RAM).
    """
    import gc
    import torch

    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Quantize layer by layer to keep peak memory low
            layers = model.model.layers
            for i in range(len(layers)):
                layers[i] = torch.quantization.quantize_dynamic(
                    layers[i], {torch.nn.Linear}, dtype=torch.qint8,
                )
                gc.collect()
            log.info(f"INT8 quantization applied to {len(layers)} language model layers")
        else:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            log.info("INT8 quantization applied to entire model")
    except Exception as e:
        log.warning(f"INT8 quantization failed, continuing without: {e}")

    return model


def unload_model(model, tokenizer) -> None:
    """Free model memory."""
    import torch
    import gc

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Model unloaded and memory freed")
