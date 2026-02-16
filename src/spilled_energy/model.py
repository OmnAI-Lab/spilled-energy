"""
Model loading utilities.

This module provides functions for loading HuggingFace models
with appropriate configurations for inference.
"""

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: str = "float16",
    trust_remote_code: bool = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace causal language model and tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B")
        device: Device to load model on ("cuda", "cpu", or None for auto)
        dtype: Model dtype ("float16", "float32", "bfloat16")
        trust_remote_code: Whether to trust remote code in model repo
        quantization_config: Quantization configuration for loading quantized models

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    # Use float32 on CPU
    if device == "cpu":
        torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
    )

    model = model.to(device)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Set to evaluation mode
    model.eval()

    return model, tokenizer
