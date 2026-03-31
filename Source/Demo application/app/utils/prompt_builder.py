"""
Prompt building utilities
Kế thừa hoàn toàn từ notebook ViMUNCH_Vistral-7B-Chat_Shot.ipynb
"""
import json
from typing import Optional, List, Dict, Any

from .config import ALLOWED_TYPES


def gold_to_annotate_json(r: Dict[str, Any]) -> Dict[str, Any]:
    """Few-shot cho bước Task 1–3: KHÔNG đưa scores để model không học tự chấm."""
    have = int(r.get("have_metaphor", 0))

    phrases = []
    for sp in (r.get("metaphor_phrases") or []):
        phrases.append({
            "phrase": sp["phrase"],
            "start": int(sp["start"]),
            "end": int(sp["end"]),
        })

    types = [t for t in (r.get("metaphor_types") or []) if t in ALLOWED_TYPES]
    interp = (r.get("interpretation") or "").strip()

    if have == 0:
        return {
            "have_metaphor": 0,
            "metaphor_phrases": [],
            "metaphor_types": [],
            "interpretation": "",
            "scores": None
        }

    return {
        "have_metaphor": 1,
        "metaphor_phrases": phrases,
        "metaphor_types": types,
        "interpretation": interp,
        "scores": None
    }


def gold_to_judge_json(r: Dict[str, Any]) -> Dict[str, Any]:
    """Few-shot cho bước Task 4: chấm GOLD interpretation."""
    gold_interp = (r.get("interpretation") or "").strip()
    if not gold_interp:
        return {"scores": None}
    return {"scores": r.get("scores", None)}


def build_prompt_annotate(
    sentence: str,
    approach: str,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build prompt cho Task 1a, 1b, 2 (nhận diện, trích xuất, phân loại)
    Mode: annotate - scores luôn null
    """
    schema_hint_annotate = {
        "have_metaphor": 0,
        "metaphor_phrases": [{"phrase": "...", "start": 0, "end": 0}],
        "metaphor_types": ["emotional"],
        "interpretation": "...",
        "scores": None
    }

    types_str = json.dumps(ALLOWED_TYPES, ensure_ascii=False)

    rules = f"""
Bạn là người gán nhãn ẩn dụ tiếng Việt. Trả về DUY NHẤT 1 JSON HỢP LỆ (không markdown, không giải thích, không lặp lại input).

ĐỊNH NGHĨA NGẮN:
Ẩn dụ là cách diễn đạt dựa trên ánh xạ khái niệm: dùng miền nguồn (cụ thể/quen thuộc) để nói miền mục tiêu (trừu tượng/khó nắm).

TASK 1a) Nhận diện ẩn dụ
- have_metaphor: 1 nếu có ít nhất 1 ẩn dụ, ngược lại 0.

TASK 1b) Trích xuất span
- metaphor_phrases: list các span chứa ẩn dụ.
  + start/end 0-based, end exclusive.
  + phrase PHẢI khớp đúng sentence[start:end].
  + Chỉ chọn phần mang nghĩa ẩn dụ (không chọn phần thuần nghĩa đen).
  + Ưu tiên span "đủ ý" tạo nên ẩn dụ.

TASK 2) Phân loại ẩn dụ (multi-label, cấp câu)
- metaphor_types: chọn 0 hoặc nhiều nhãn trong đúng tập sau: {types_str}
Gợi ý cực ngắn:
- structural: hiểu A theo cấu trúc của B
- orientational: dùng hướng không gian (lên/xuống, vào/ra...) để biểu đạt trạng thái/giá trị
- ontological: xem khái niệm trừu tượng như vật thể/sinh thể có thể "tương tác"
- emotional: dùng hình ảnh để gợi/biểu đạt cảm xúc
- cultural_folklore: gắn văn hóa dân gian/điển cố/ca dao-tục ngữ/biểu tượng Việt

QUY TẮC:
- interpretation: để rỗng "" (Task 3 sẽ xử lý riêng)
- scores: LUÔN = null

QUY TẮC NULL (bắt buộc):
- Nếu have_metaphor=0:
  metaphor_phrases=[], metaphor_types=[], interpretation="", scores=null

BẮT BUỘC FORMAT:
- Chỉ xuất 1 JSON object, bắt đầu bằng {{ và kết thúc bằng }}.
- Dùng nháy kép " cho mọi key và string value.
- metaphor_phrases là LIST object đúng dạng:
  {{"phrase":"...","start":0,"end":0}}

Schema mẫu: {json.dumps(schema_hint_annotate, ensure_ascii=False)}
""".strip()

    fewshot_block = ""
    if approach == "few_shot_5" and fewshot_examples:
        ex_parts = []
        for ex in fewshot_examples:
            ex_out = gold_to_annotate_json(ex)
            ex_out["interpretation"] = ""  # Task 1-2 không có interpretation
            ex_parts.append("Ví dụ\nCâu: " + ex["sentence"])
            ex_parts.append("JSON:\n" + json.dumps(ex_out, ensure_ascii=False))
        fewshot_block = "\n\n".join(ex_parts)

    user_part = f"Câu: {sentence}\nJSON:"
    if fewshot_block:
        return rules + "\n\n" + fewshot_block + "\n\nBài làm\n" + user_part
    return rules + "\n\n" + user_part


def build_prompt_interpret(
    sentence: str,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    have_metaphor: int = 1,
    metaphor_phrases: Optional[List[Dict]] = None,
    metaphor_types: Optional[List[str]] = None,
) -> str:
    """
    Build prompt cho Task 3 (diễn giải) - LUÔN dùng Few-shot 5
    """
    schema_hint = {
        "interpretation": "..."
    }

    # Context từ Task 1-2
    context_info = ""
    if have_metaphor == 1:
        if metaphor_phrases:
            spans_str = ", ".join([f'"{p["phrase"]}"' for p in metaphor_phrases])
            context_info += f"\nCác span ẩn dụ đã trích xuất: {spans_str}"
        if metaphor_types:
            types_str = ", ".join(metaphor_types)
            context_info += f"\nLoại ẩn dụ: {types_str}"

    rules = f"""
Bạn là người diễn giải ẩn dụ tiếng Việt. Trả về DUY NHẤT 1 JSON HỢP LỆ (không markdown, không giải thích, không lặp lại input).

NHIỆM VỤ (TASK 3): Diễn giải ẩn dụ
- interpretation: viết lại cho tường minh hơn (KHÔNG chép lại câu gốc) nhưng KHÔNG đổi nghĩa, KHÔNG thêm ý và KHÔNG bớt ý.
- Viết tiếng Việt tự nhiên, ngắn gọn.
- KHÔNG dùng dấu nháy kép " trong interpretation (nếu cần trích dẫn dùng nháy đơn '...').
{context_info}

BẮT BUỘC FORMAT:
- Chỉ xuất 1 JSON object với key "interpretation".
- Dùng nháy kép " cho key và string value.

Schema mẫu: {json.dumps(schema_hint, ensure_ascii=False)}
""".strip()

    fewshot_block = ""
    if fewshot_examples:
        ex_parts = []
        for ex in fewshot_examples:
            if ex.get("have_metaphor", 0) == 1 and ex.get("interpretation"):
                ex_parts.append("Ví dụ\nCâu: " + ex["sentence"])
                ex_parts.append("JSON:\n" + json.dumps({"interpretation": ex["interpretation"]}, ensure_ascii=False))
        if ex_parts:
            fewshot_block = "\n\n".join(ex_parts)

    user_part = f"Câu: {sentence}\nJSON:"
    if fewshot_block:
        return rules + "\n\n" + fewshot_block + "\n\nBài làm\n" + user_part
    return rules + "\n\n" + user_part


def build_prompt_judge(
    sentence: str,
    interpretation: str,
    approach: str = "zero_shot",
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build prompt cho Task 4 (chấm điểm diễn giải)
    """
    schema_hint_judge = {
        "scores": {
            "accuracy": 0, "clarity": 0, "naturalness": 0,
            "meaning": 0, "implication": 0, "modality": 0, "syntax": 0, "context": 0,
            "overall": 0.0, "quality": 0.0
        }
    }

    interpretation = (interpretation or "").strip()

    rules = f"""
Bạn là người chấm điểm câu diễn giải (paraphrase) tiếng Việt. Trả về DUY NHẤT 1 JSON HỢP LỆ (không markdown, không giải thích, không lặp lại input).

NHIỆM VỤ (TASK 4):
- Chỉ chấm điểm cho INTERPRETATION được cung cấp. KHÔNG viết lại, KHÔNG sửa câu.
- Nếu INTERPRETATION rỗng => scores=null.

Chấm 8 tiêu chí (0–4):
A) Chất lượng diễn giải: accuracy, clarity, naturalness
B) Tương đồng ngữ nghĩa: meaning, implication, modality, syntax, context

Tính 2 điểm tổng (làm tròn 1 chữ số thập phân):
- overall = (3*accuracy + 2*clarity + 1*naturalness) / 6
- quality = meaning*0.3 + implication*0.25 + modality*0.25 + syntax*0.1 + context*0.1

BẮT BUỘC FORMAT:
- Chỉ xuất 1 JSON object, bắt đầu bằng {{ và kết thúc bằng }}.
- Dùng nháy kép " cho key và string value.
- Output chỉ theo schema: {json.dumps(schema_hint_judge, ensure_ascii=False)}
""".strip()

    fewshot_block = ""
    if approach == "few_shot_5" and fewshot_examples:
        ex_parts = []
        for ex in fewshot_examples:
            ex_out = gold_to_judge_json(ex)
            gi = (ex.get("interpretation") or "").strip() or "<EMPTY>"
            ex_parts.append("Ví dụ\nCâu: " + ex["sentence"])
            ex_parts.append("INTERPRETATION: " + gi)
            ex_parts.append("JSON:\n" + json.dumps(ex_out, ensure_ascii=False))
        fewshot_block = "\n\n".join(ex_parts)

    user_part = (
        f"Câu: {sentence}\n"
        f"INTERPRETATION: {interpretation if interpretation else '<EMPTY>'}\n"
        f"JSON:"
    )
    if fewshot_block:
        return rules + "\n\n" + fewshot_block + "\n\nBài làm\n" + user_part
    return rules + "\n\n" + user_part


# ====== Legacy function for compatibility ======
def build_prompt(
    sentence: str,
    approach: str,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    *,
    mode: str = "annotate",
    gold_interpretation: Optional[str] = None
) -> str:
    """
    Legacy function - giữ để tương thích với code cũ.
    annotate: Task 1–3 (model tự diễn giải), scores luôn null
    judge:    Task 4 (chấm GOLD interpretation), output chỉ có scores
    """
    types_str = json.dumps(ALLOWED_TYPES, ensure_ascii=False)

    schema_hint_annotate = {
        "have_metaphor": 0,
        "metaphor_phrases": [{"phrase": "...", "start": 0, "end": 0}],
        "metaphor_types": ["emotional"],
        "interpretation": "...",
        "scores": None
    }

    schema_hint_judge = {
        "scores": {
            "accuracy": 0, "clarity": 0, "naturalness": 0,
            "meaning": 0, "implication": 0, "modality": 0, "syntax": 0, "context": 0,
            "overall": 0.0, "quality": 0.0
        }
    }

    if mode == "annotate":
        rules = f"""
Bạn là người gán nhãn ẩn dụ tiếng Việt. Trả về DUY NHẤT 1 JSON HỢP LỆ (không markdown, không giải thích, không lặp lại input).

ĐỊNH NGHĨA NGẮN:
Ẩn dụ là cách diễn đạt dựa trên ánh xạ khái niệm: dùng miền nguồn (cụ thể/quen thuộc) để nói miền mục tiêu (trừu tượng/khó nắm).

TASK 1) Nhận diện + trích xuất span
- have_metaphor: 1 nếu có ít nhất 1 ẩn dụ, ngược lại 0.
- metaphor_phrases: list các span chứa ẩn dụ.
  + start/end 0-based, end exclusive.
  + phrase PHẢI khớp đúng sentence[start:end].
  + Chỉ chọn phần mang nghĩa ẩn dụ (không chọn phần thuần nghĩa đen).
  + Ưu tiên span "đủ ý" tạo nên ẩn dụ.

TASK 2) Phân loại ẩn dụ (multi-label, cấp câu)
- metaphor_types: chọn 0 hoặc nhiều nhãn trong đúng tập sau: {types_str}
Gợi ý cực ngắn:
- structural: hiểu A theo cấu trúc của B
- orientational: dùng hướng không gian (lên/xuống, vào/ra...) để biểu đạt trạng thái/giá trị
- ontological: xem khái niệm trừu tượng như vật thể/sinh thể có thể "tương tác"
- emotional: dùng hình ảnh để gợi/biểu đạt cảm xúc
- cultural_folklore: gắn văn hóa dân gian/điển cố/ca dao-tục ngữ/biểu tượng Việt

TASK 3) Diễn giải (BẠN TỰ VIẾT)
- interpretation: viết lại cho tường minh hơn (KHÔNG chép lại câu gốc) nhưng KHÔNG đổi nghĩa, KHÔNG thêm ý và KHÔNG bớt ý.
- Viết tiếng Việt tự nhiên, ngắn gọn.
- KHÔNG dùng dấu nháy kép " trong interpretation (nếu cần trích dẫn dùng nháy đơn '...').

QUY TẮC NULL (bắt buộc):
- Nếu have_metaphor=0:
  metaphor_phrases=[], metaphor_types=[], interpretation="", scores=null
- Nếu have_metaphor=1:
  cố gắng điền metaphor_phrases và metaphor_types; interpretation không rỗng.
  scores LUÔN = null (không tự chấm điểm ở bước này).

BẮT BUỘC FORMAT:
- Chỉ xuất 1 JSON object, bắt đầu bằng {{ và kết thúc bằng }}.
- Dùng nháy kép " cho mọi key và string value.
- metaphor_phrases là LIST object đúng dạng:
  {{"phrase":"...","start":0,"end":0}}

Schema mẫu: {json.dumps(schema_hint_annotate, ensure_ascii=False)}
""".strip()

        fewshot_block = ""
        if approach == "few_shot_5" and fewshot_examples:
            ex_parts = []
            for ex in fewshot_examples:
                ex_out = gold_to_annotate_json(ex)
                ex_parts.append("Ví dụ\nCâu: " + ex["sentence"])
                ex_parts.append("JSON:\n" + json.dumps(ex_out, ensure_ascii=False))
            fewshot_block = "\n\n".join(ex_parts)

        user_part = f"Câu: {sentence}\nJSON:"
        if fewshot_block:
            return rules + "\n\n" + fewshot_block + "\n\nBài làm\n" + user_part
        return rules + "\n\n" + user_part

    # JUDGE MODE (Task 4)
    gold_interpretation = (gold_interpretation or "").strip()

    rules = f"""
Bạn là người chấm điểm câu diễn giải (paraphrase) tiếng Việt. Trả về DUY NHẤT 1 JSON HỢP LỆ (không markdown, không giải thích, không lặp lại input).

NHIỆM VỤ (TASK 4):
- Chỉ chấm điểm cho GOLD_INTERPRETATION được cung cấp. KHÔNG viết lại, KHÔNG sửa câu.
- Nếu GOLD_INTERPRETATION rỗng => scores=null.

Chấm 8 tiêu chí (0–4):
A) Chất lượng diễn giải: accuracy, clarity, naturalness
B) Tương đồng ngữ nghĩa: meaning, implication, modality, syntax, context

Tính 2 điểm tổng (làm tròn 1 chữ số thập phân):
- overall = (3*accuracy + 2*clarity + 1*naturalness) / 6
- quality = meaning*0.3 + implication*0.25 + modality*0.25 + syntax*0.1 + context*0.1

BẮT BUỘC FORMAT:
- Chỉ xuất 1 JSON object, bắt đầu bằng {{ và kết thúc bằng }}.
- Dùng nháy kép " cho key và string value.
- Output chỉ theo schema: {json.dumps(schema_hint_judge, ensure_ascii=False)}
""".strip()

    fewshot_block = ""
    if approach == "few_shot_5" and fewshot_examples:
        ex_parts = []
        for ex in fewshot_examples:
            ex_out = gold_to_judge_json(ex)
            gi = (ex.get("interpretation") or "").strip() or "<EMPTY>"
            ex_parts.append("Ví dụ\nCâu: " + ex["sentence"])
            ex_parts.append("GOLD_INTERPRETATION: " + gi)
            ex_parts.append("JSON:\n" + json.dumps(ex_out, ensure_ascii=False))
        fewshot_block = "\n\n".join(ex_parts)

    user_part = (
        f"Câu: {sentence}\n"
        f"GOLD_INTERPRETATION: {gold_interpretation if gold_interpretation else '<EMPTY>'}\n"
        f"JSON:"
    )
    if fewshot_block:
        return rules + "\n\n" + fewshot_block + "\n\nBài làm\n" + user_part
    return rules + "\n\n" + user_part
