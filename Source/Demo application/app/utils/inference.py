"""
Inference pipeline for ViMUNCH Demo
Kế thừa hoàn toàn từ notebook ViMUNCH_Vistral-7B-Chat_Shot.ipynb

Pipeline:
- Task 1a, 1b, 2: Zero-shot / Fine-tuned
- Task 3 (Diễn giải): Few-shot 5
- Task 4 (Chấm điểm): Zero-shot / Fine-tuned
"""
import torch
from typing import Optional, List, Dict, Any, Tuple

from .config import (
    MAX_NEW_TOKENS,
    RETRY_MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
)
from .model_loader import to_chat_prompt
from .prompt_builder import (
    build_prompt_annotate,
    build_prompt_interpret,
    build_prompt_judge,
)
from .json_parser import (
    extract_json,
    normalize_annotate,
    normalize_interpret,
    normalize_judge,
    align_phrases_by_phrase,
    fill_overall_quality_if_missing,
    validate_annotate,
    validate_judge,
)


@torch.no_grad()
def generate_single(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: Optional[int] = None
) -> str:
    """
    Generate output cho 1 prompt
    Kế thừa từ notebook generate_batch()
    """
    mn = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

    chat_prompt = to_chat_prompt(tokenizer, prompt)
    
    inputs = tokenizer(
        chat_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=4096
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": mn,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if DO_SAMPLE:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = TEMPERATURE if TEMPERATURE > 0 else 0.1
    else:
        gen_kwargs["do_sample"] = False

    out = model.generate(**inputs, **gen_kwargs)

    attn = inputs["attention_mask"]
    prompt_len = int(attn[0].sum().item())
    gen_ids = out[0, prompt_len:]
    
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_task_1_2(
    model,
    tokenizer,
    sentence: str,
    approach: str = "zero_shot",
    fewshot_examples: Optional[List[Dict]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run Task 1a (nhận diện), 1b (trích xuất), 2 (phân loại)
    
    Args:
        model: Model đã load
        tokenizer: Tokenizer
        sentence: Câu cần phân tích
        approach: "zero_shot" hoặc "few_shot_5" (thường dùng zero_shot/ft)
        fewshot_examples: Ví dụ few-shot nếu cần
    
    Returns:
        (result_dict, meta_dict)
    """
    prompt = build_prompt_annotate(sentence, approach, fewshot_examples)
    raw = generate_single(model, tokenizer, prompt)
    
    obj, perr = extract_json(raw)
    retried = False
    
    # Retry nếu parse fail
    if obj is None and perr in ["no_json_object", "json_load_error: cannot_parse_any_object"]:
        raw2 = generate_single(model, tokenizer, prompt, max_new_tokens=RETRY_MAX_NEW_TOKENS)
        obj2, perr2 = extract_json(raw2)
        if obj2 is not None:
            obj, perr, raw = obj2, perr2, raw2
            retried = True
    
    meta = {
        "raw": raw,
        "parse_error": perr,
        "retried": retried,
        "valid": False,
        "validate_error": None,
    }
    
    if obj is None:
        result = {
            "have_metaphor": 0,
            "metaphor_phrases": [],
            "metaphor_types": [],
            "interpretation": "",
            "scores": None,
        }
        return result, meta
    
    # Normalize và validate
    obj = normalize_annotate(obj)
    obj = align_phrases_by_phrase(obj, sentence)
    ok, verr = validate_annotate(obj, sentence)
    
    meta["valid"] = ok
    meta["validate_error"] = verr
    
    return obj, meta


def run_task_3(
    model,
    tokenizer,
    sentence: str,
    have_metaphor: int,
    metaphor_phrases: Optional[List[Dict]] = None,
    metaphor_types: Optional[List[str]] = None,
    fewshot_examples: Optional[List[Dict]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Run Task 3 (diễn giải) - LUÔN dùng Few-shot 5
    
    Args:
        model: Model đã load
        tokenizer: Tokenizer
        sentence: Câu cần phân tích
        have_metaphor: Kết quả từ Task 1a
        metaphor_phrases: Kết quả từ Task 1b
        metaphor_types: Kết quả từ Task 2
        fewshot_examples: Ví dụ few-shot (bắt buộc)
    
    Returns:
        (interpretation, meta_dict)
    """
    if have_metaphor == 0:
        return "", {"raw": None, "skipped": True}
    
    prompt = build_prompt_interpret(
        sentence, 
        fewshot_examples,
        have_metaphor,
        metaphor_phrases,
        metaphor_types
    )
    raw = generate_single(model, tokenizer, prompt)
    
    obj, perr = extract_json(raw)
    retried = False
    
    # Retry nếu parse fail
    if obj is None and perr in ["no_json_object", "json_load_error: cannot_parse_any_object"]:
        raw2 = generate_single(model, tokenizer, prompt, max_new_tokens=RETRY_MAX_NEW_TOKENS)
        obj2, perr2 = extract_json(raw2)
        if obj2 is not None:
            obj, perr, raw = obj2, perr2, raw2
            retried = True
    
    meta = {
        "raw": raw,
        "parse_error": perr,
        "retried": retried,
        "skipped": False,
    }
    
    if obj is None:
        return "", meta
    
    obj = normalize_interpret(obj)
    interpretation = obj.get("interpretation", "")
    
    return interpretation, meta


def run_task_4(
    model,
    tokenizer,
    sentence: str,
    interpretation: str,
    approach: str = "zero_shot",
    fewshot_examples: Optional[List[Dict]] = None
) -> Tuple[Optional[Dict], Dict[str, Any]]:
    """
    Run Task 4 (chấm điểm diễn giải)
    
    Args:
        model: Model đã load
        tokenizer: Tokenizer
        sentence: Câu gốc
        interpretation: Câu diễn giải từ Task 3
        approach: "zero_shot" hoặc "few_shot_5" (thường dùng zero_shot/ft)
        fewshot_examples: Ví dụ few-shot nếu cần
    
    Returns:
        (scores_dict, meta_dict)
    """
    if not interpretation or not interpretation.strip():
        return None, {"raw": None, "skipped": True}
    
    prompt = build_prompt_judge(sentence, interpretation, approach, fewshot_examples)
    raw = generate_single(model, tokenizer, prompt)
    
    obj, perr = extract_json(raw)
    retried = False
    
    # Retry nếu parse fail
    if obj is None and perr in ["no_json_object", "json_load_error: cannot_parse_any_object"]:
        raw2 = generate_single(model, tokenizer, prompt, max_new_tokens=RETRY_MAX_NEW_TOKENS)
        obj2, perr2 = extract_json(raw2)
        if obj2 is not None:
            obj, perr, raw = obj2, perr2, raw2
            retried = True
    
    meta = {
        "raw": raw,
        "parse_error": perr,
        "retried": retried,
        "valid": False,
        "validate_error": None,
        "skipped": False,
    }
    
    if obj is None:
        return None, meta
    
    # Normalize và validate
    obj = normalize_judge(obj)
    obj = fill_overall_quality_if_missing(obj)
    ok, verr = validate_judge(obj)
    
    meta["valid"] = ok
    meta["validate_error"] = verr
    
    scores = obj.get("scores")
    return scores, meta


def run_full_pipeline(
    model_base,
    tokenizer_base,
    sentence: str,
    fewshot_examples: List[Dict],
    model_ft=None,
    tokenizer_ft=None,
    use_ft_for_task_1_2: bool = False,
    use_ft_for_task_4: bool = False,
    skip_task_4: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run full pipeline cho 1 câu:
    - Task 1a, 1b, 2: Zero-shot (base) hoặc Fine-tuned
    - Task 3: Few-shot 5 (base model)
    - Task 4: Zero-shot (base) hoặc Fine-tuned
    
    Args:
        model_base: Base model (Vistral-7B-Chat)
        tokenizer_base: Tokenizer cho base model
        sentence: Câu cần phân tích
        fewshot_examples: 5 ví dụ few-shot cho Task 3
        model_ft: Fine-tuned model (optional)
        tokenizer_ft: Tokenizer cho FT model (optional)
        use_ft_for_task_1_2: Dùng FT model cho Task 1-2
        use_ft_for_task_4: Dùng FT model cho Task 4
        progress_callback: Callback để update progress (optional)
    
    Returns:
        Dict chứa kết quả và metadata
    """
    result = {
        "sentence": sentence,
        "task_1a": {"have_metaphor": 0},
        "task_1b": {"metaphor_phrases": []},
        "task_2": {"metaphor_types": []},
        "task_3": {"interpretation": ""},
        "task_4": {"scores": None},
        "meta": {}
    }
    
    # ===== Task 1a, 1b, 2 =====
    if progress_callback:
        progress_callback("Đang chạy Task 1a, 1b, 2 (Nhận diện, Trích xuất, Phân loại)...")
    
    if use_ft_for_task_1_2 and model_ft is not None:
        model_12, tok_12 = model_ft, tokenizer_ft
        approach_12 = "zero_shot"  # FT model không cần few-shot
    else:
        model_12, tok_12 = model_base, tokenizer_base
        approach_12 = "zero_shot"
    
    annotate_result, annotate_meta = run_task_1_2(
        model_12, tok_12, sentence, approach_12
    )
    
    result["task_1a"]["have_metaphor"] = annotate_result.get("have_metaphor", 0)
    result["task_1b"]["metaphor_phrases"] = annotate_result.get("metaphor_phrases", [])
    result["task_2"]["metaphor_types"] = annotate_result.get("metaphor_types", [])
    result["meta"]["task_1_2"] = annotate_meta
    
    # ===== Task 3: Few-shot 5 (luôn dùng base model) =====
    if progress_callback:
        progress_callback("Đang chạy Task 3 (Diễn giải) với Few-shot 5...")
    
    interpretation, interpret_meta = run_task_3(
        model_base,
        tokenizer_base,
        sentence,
        result["task_1a"]["have_metaphor"],
        result["task_1b"]["metaphor_phrases"],
        result["task_2"]["metaphor_types"],
        fewshot_examples
    )
    
    result["task_3"]["interpretation"] = interpretation
    result["meta"]["task_3"] = interpret_meta
    
    # ===== Task 4: Zero-shot / Fine-tuned =====
    if not skip_task_4:
        if progress_callback:
            progress_callback("Đang chạy Task 4 (Chấm điểm)...")
        
        if use_ft_for_task_4 and model_ft is not None:
            model_4, tok_4 = model_ft, tokenizer_ft
            approach_4 = "zero_shot"
        else:
            model_4, tok_4 = model_base, tokenizer_base
            approach_4 = "zero_shot"
        
        scores, judge_meta = run_task_4(
            model_4, tok_4, sentence, interpretation, approach_4
        )
        
        result["task_4"]["scores"] = scores
        result["meta"]["task_4"] = judge_meta
    else:
        result["meta"]["task_4"] = {"skipped": True}
    
    if progress_callback:
        progress_callback("Hoàn thành!")
    
    return result
