import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from typing import Tuple, Optional

def load_model_and_tokenizer(
    model_name: str,
    device: str = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the language model and tokenizer.

    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on (auto-detected if None)
        load_in_4bit: Use 4-bit quantization (recommended for 70B models)
        load_in_8bit: Use 8-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model '{model_name}' on device '{device}'...")

    # Prepare quantization config if needed
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Using 4-bit quantization")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        print("Using 8-bit quantization")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Try flash_attention_2 first, fall back to eager if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception as flash_err:
            print(f"FlashAttention2 not available, falling back to eager attention: {flash_err}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                **model_kwargs,
            )

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        print(f"Model loaded successfully on {device}")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Convenience function to detect model type from model name
def detect_model_type(model_name: str) -> str:
    """
    Detect model type from model name.

    Args:
        model_name: Model name or path

    Returns:
        "llama" or "gemma"
    """
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return "llama"
    elif "gemma" in model_name_lower:
        return "gemma"
    elif "qwen" in model_name_lower:
        return "qwen"
    else:
        # Default to llama for unknown models
        print(f"Warning: Could not detect model type from '{model_name}', defaulting to 'llama'")
        return "llama"
