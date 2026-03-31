"""
Model loading utilities
Kế thừa hoàn toàn từ notebook ViMUNCH_Vistral-7B-Chat_Shot.ipynb
"""
import os
import sys
import torch
from typing import Optional, Tuple

from .config import (
    MODEL_NAME, 
    FT_MODEL_PATH, 
    TRUST_REMOTE_CODE, 
    DTYPE,
    HF_TOKEN,
)

# Helper để log ra stderr
def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# Global model cache
_model_cache = {}


def _login_huggingface():
    """Login to HuggingFace Hub"""
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        _log(f"Warning: Could not login to HuggingFace: {e}")


def load_base_model(device_map: str = "auto"):
    """
    Load base Vistral model (cho zero-shot và few-shot)
    Sử dụng cache để tránh load lại nhiều lần
    """
    cache_key = f"base_{MODEL_NAME}"
    
    if cache_key in _model_cache:
        _log(f"   [Cache hit] Base model already loaded")
        return _model_cache[cache_key]
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Login to HuggingFace
    _log(f"   [1/4] Logging in to HuggingFace...")
    _login_huggingface()
    
    _log(f"   [2/4] Loading tokenizer from {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=TRUST_REMOTE_CODE
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _log(f"   [2/4] ✓ Tokenizer loaded")

    _log(f"   [3/4] Loading model weights from {MODEL_NAME}...")
    _log(f"         (This may take several minutes...)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=device_map,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    _log(f"   [3/4] ✓ Model weights loaded")

    _log(f"   [4/4] ✓ Model ready on device: {next(model.parameters()).device}")
    
    _model_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


def load_finetuned_model(ft_path: Optional[str] = None, device_map: str = "auto"):
    """
    Load fine-tuned model với LoRA adapter
    Sử dụng cache để tránh load lại nhiều lần
    """
    ft_path = ft_path or FT_MODEL_PATH
    cache_key = f"ft_{ft_path}"
    
    if cache_key in _model_cache:
        _log(f"   [Cache hit] Fine-tuned model already loaded")
        return _model_cache[cache_key]
    
    if not os.path.exists(ft_path):
        raise FileNotFoundError(f"Fine-tuned model not found at: {ft_path}")
    
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    _log(f"   [1/3] Loading LoRA adapter from {ft_path}...")
    _log(f"         (This may take a few minutes...)")
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        ft_path,
        device_map=device_map,
        torch_dtype=DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.eval()
    _log(f"   [1/3] ✓ LoRA adapter loaded")

    _log(f"   [2/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ft_path, 
        use_fast=True, 
        trust_remote_code=TRUST_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _log(f"   [2/3] ✓ Tokenizer loaded")

    _log(f"   [3/3] ✓ Fine-tuned model ready on device: {next(model.parameters()).device}")
    
    _model_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


def get_model_and_tokenizer(use_finetuned: bool = False, ft_path: Optional[str] = None):
    """
    Get model và tokenizer theo config
    
    Args:
        use_finetuned: True để dùng model fine-tuned, False để dùng base model
        ft_path: Đường dẫn custom đến fine-tuned model
    
    Returns:
        (model, tokenizer)
    """
    if use_finetuned:
        return load_finetuned_model(ft_path)
    else:
        return load_base_model()


def to_chat_prompt(tokenizer, text_prompt: str) -> str:
    """
    Convert text prompt sang chat format
    Kế thừa từ notebook
    """
    messages = [
        {"role": "system", "content": "Chỉ được xuất ra JSON hợp lệ, không thêm bất kỳ văn bản nào khác."},
        {"role": "user", "content": text_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            pass

    # fallback Llama-style
    sys = "Chỉ được xuất ra JSON hợp lệ, không thêm bất kỳ văn bản nào khác."
    return f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{text_prompt} [/INST]"


def clear_model_cache():
    """Clear model cache để giải phóng VRAM"""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache()
    print("✅ Model cache cleared")


def check_gpu_available() -> Tuple[bool, str]:
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        return False, "No GPU available - running on CPU"


def get_model_info() -> dict:
    """Get information about loaded models"""
    info = {
        "base_model": MODEL_NAME,
        "ft_model_path": FT_MODEL_PATH,
        "ft_model_exists": os.path.exists(FT_MODEL_PATH),
        "dtype": str(DTYPE),
        "cached_models": list(_model_cache.keys()),
    }
    
    gpu_available, gpu_info = check_gpu_available()
    info["gpu_available"] = gpu_available
    info["gpu_info"] = gpu_info
    
    return info
