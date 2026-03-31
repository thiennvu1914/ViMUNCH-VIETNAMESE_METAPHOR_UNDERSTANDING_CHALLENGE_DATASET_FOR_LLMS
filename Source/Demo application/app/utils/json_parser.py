"""
JSON extraction & validation utilities
Kế thừa hoàn toàn từ notebook ViMUNCH_Vistral-7B-Chat_Shot.ipynb
"""
import json
import re
import ast
from typing import Any, Dict, List, Tuple, Optional

from .config import ALLOWED_TYPES, SCORE_KEYS


# =========================
# JSON extraction utilities
# =========================
def _strip_noise(text: str) -> str:
    t = text.replace("```json", "").replace("```", "")
    t = t.replace("<|im_start|>", "").replace("<|im_end|>", "")
    t = t.replace("<s>", "").replace("</s>", "")
    t = t.replace("[INST]", "").replace("[/INST]", "")
    
    # ưu tiên bắt JSON output cuối (tránh model echo schema/prompt)
    m = list(re.finditer(r'\{\s*"(have_metaphor|scores)"\s*:', t))
    if m:
        return t[m[-1].start():]

    j = t.rfind("{")
    return t[j:] if j != -1 else t


def _find_top_level_json_objects(s: str) -> List[str]:
    objs, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(s[start:i+1])
                    start = None
    return objs


def _repair_span_double_braces(js: str) -> str:
    s = js

    # sửa lỗi }} trước , hoặc ]
    s = re.sub(r"\}\}(?=\s*[,]])", "}", s)

    # sửa lỗi đóng list rồi mới đóng object:  ... ]} , {  ->  ... } , {
    s = re.sub(r"\]\s*\}(?=\s*,\s*\{)", "}", s)

    # (phòng trường hợp ngược lại) ... }], {  ->  ... }, {
    s = re.sub(r"\}\s*\](?=\s*,\s*\{)", "}", s)

    #  bỏ 1 dấu '}' cuối nếu thừa đúng 1 cái
    if s.count("}") == s.count("{") + 1 and s.rstrip().endswith("}"):
        s = s.rstrip()[:-1]

    return s


def _repair_py_literals(js: str) -> str:
    s = re.sub(r"\bNone\b", "null", js)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _parse_one_obj(obj_str: str) -> Optional[Dict[str, Any]]:
    st = _repair_span_double_braces(obj_str.strip())

    try:
        obj = json.loads(st)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    st2 = _repair_py_literals(st)
    try:
        obj = json.loads(st2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    try:
        py = st
        py = re.sub(r"\bnull\b", "None", py)
        py = re.sub(r"\btrue\b", "True", py)
        py = re.sub(r"\bfalse\b", "False", py)
        obj = ast.literal_eval(py)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _balance_json_tail(s: str) -> str:
    # thêm ] và } còn thiếu nếu model bị cắt cụt ở cuối
    ob, cb = s.count("{"), s.count("}")
    osq, csq = s.count("["), s.count("]")
    if osq > csq:
        s += "]" * (osq - csq)
    if ob > cb:
        s += "}" * (ob - cb)
    return s


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Trích xuất JSON từ output của model.
    Returns: (parsed_dict, error_message)
    """
    t = _strip_noise(text)
    obj_strs = _find_top_level_json_objects(t)
    if not obj_strs:
        i = t.find("{")
        if i != -1:
            cand = _balance_json_tail(t[i:].strip())
            obj = _parse_one_obj(cand)
            if obj is not None:
                return obj, None
        return None, "no_json_object"

    merged, parsed_any = {}, False
    for s in obj_strs:
        obj = _parse_one_obj(s)
        if obj is not None:
            merged.update(obj)
            parsed_any = True

    if not parsed_any:
        return None, "json_load_error: cannot_parse_any_object"

    return merged, None


# =========================
# normalize + deterministic repairs
# =========================
def normalize_annotate(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize kết quả annotate (Task 1a, 1b, 2)"""
    out = dict(obj)

    # map key cũ -> key mới (phòng khi model lỡ trả spans/types)
    if "metaphor_phrases" not in out and "spans" in out:
        mp = []
        for sp in (out.get("spans") or []):
            if isinstance(sp, dict):
                mp.append({
                    "phrase": str(sp.get("phrase", "")),
                    "start": sp.get("start", 0),
                    "end": sp.get("end", 0)
                })
        out["metaphor_phrases"] = mp
    if "metaphor_types" not in out and "types" in out:
        out["metaphor_types"] = out.get("types")

    # have_metaphor
    hm = out.get("have_metaphor", 0)
    try:
        hm = int(hm)
    except Exception:
        hm = 0
    hm = 1 if hm == 1 else 0
    out["have_metaphor"] = hm

    # defaults
    if "metaphor_phrases" not in out or not isinstance(out["metaphor_phrases"], list):
        out["metaphor_phrases"] = []
    if "metaphor_types" not in out or not isinstance(out["metaphor_types"], list):
        out["metaphor_types"] = []
    if "interpretation" not in out or not isinstance(out["interpretation"], str):
        out["interpretation"] = ""

    # annotate mode: scores luôn null
    out["scores"] = None

    # have_metaphor=0 => ép rỗng
    if hm == 0:
        out["metaphor_phrases"] = []
        out["metaphor_types"] = []
        out["interpretation"] = ""

    # clean phrases
    clean = []
    for sp in out["metaphor_phrases"]:
        if not isinstance(sp, dict):
            continue
        ph = str(sp.get("phrase", ""))
        if not ph:
            continue
        try:
            st = int(sp.get("start", 0))
            ed = int(sp.get("end", 0))
        except Exception:
            st, ed = 0, 0
        clean.append({"phrase": ph, "start": st, "end": ed})
    out["metaphor_phrases"] = clean

    # clean types theo whitelist
    out["metaphor_types"] = [t for t in out["metaphor_types"] if t in ALLOWED_TYPES]

    return out


def normalize_interpret(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize kết quả interpret (Task 3) - chỉ lấy interpretation"""
    out = dict(obj)
    
    if "interpretation" not in out or not isinstance(out["interpretation"], str):
        out["interpretation"] = ""
    
    return out


def align_phrases_by_phrase(pred: Dict[str, Any], sentence: str) -> Dict[str, Any]:
    """
    Deterministic repair (công bằng):
    - Nếu start/end không khớp substring nhưng phrase xuất hiện trong sentence -> set lại start/end theo find().
    - Nếu phrase không xuất hiện -> drop span đó (không fuzzy để tránh bias).
    """
    if not isinstance(pred, dict):
        return pred
    phs = pred.get("metaphor_phrases", [])
    if not isinstance(phs, list):
        pred["metaphor_phrases"] = []
        return pred

    fixed = []
    for sp in phs:
        if not isinstance(sp, dict):
            continue
        ph = str(sp.get("phrase", ""))
        if not ph:
            continue

        st = sp.get("start", None)
        ed = sp.get("end", None)

        # case 1: đã khớp substring
        if isinstance(st, int) and isinstance(ed, int) and 0 <= st <= ed <= len(sentence) and sentence[st:ed] == ph:
            fixed.append({"phrase": ph, "start": st, "end": ed})
            continue

        # case 2: align bằng find()
        idx = sentence.find(ph)
        if idx != -1:
            fixed.append({"phrase": ph, "start": idx, "end": idx + len(ph)})
            continue

        # case 3: không thấy phrase -> drop
        pass

    pred["metaphor_phrases"] = fixed
    return pred


def normalize_judge(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize kết quả judge (Task 4)"""
    out = dict(obj)

    # wrap scores nếu model trả phẳng
    if "scores" not in out:
        if any(k in out for k in SCORE_KEYS):
            out = {"scores": {k: out.get(k) for k in SCORE_KEYS}}
        else:
            out = {"scores": None}

    sc = out.get("scores", None)
    if sc is None:
        out["scores"] = None
        return out

    if not isinstance(sc, dict):
        out["scores"] = None
        return out

    out["scores"] = {k: sc.get(k) for k in SCORE_KEYS}
    return out


def _round1(x: float) -> float:
    return round(float(x), 1)


def fill_overall_quality_if_missing(j: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nếu model quên overall/quality (null) nhưng có đủ điểm con -> tính lại theo công thức cố định.
    """
    if not isinstance(j, dict) or "scores" not in j:
        return j
    sc = j["scores"]
    if sc is None or not isinstance(sc, dict):
        return j

    # cast string -> float nếu cần
    for k in SCORE_KEYS:
        if k in sc and isinstance(sc[k], str):
            try:
                sc[k] = float(sc[k])
            except Exception:
                pass

    # overall
    if sc.get("overall") is None:
        a, c, n = sc.get("accuracy"), sc.get("clarity"), sc.get("naturalness")
        if isinstance(a, (int, float)) and isinstance(c, (int, float)) and isinstance(n, (int, float)):
            sc["overall"] = _round1((3*a + 2*c + 1*n)/6)

    # quality
    if sc.get("quality") is None:
        m = sc.get("meaning")
        imp = sc.get("implication")
        mod = sc.get("modality")
        syn = sc.get("syntax")
        ctx = sc.get("context")
        if all(isinstance(x, (int, float)) for x in [m, imp, mod, syn, ctx]):
            sc["quality"] = _round1(m*0.3 + imp*0.25 + mod*0.25 + syn*0.1 + ctx*0.1)

    j["scores"] = sc
    return j


# =========================
# validate
# =========================
def validate_annotate(obj: Any, sentence: str) -> Tuple[bool, Optional[str]]:
    """Validate kết quả annotate"""
    need = ["have_metaphor", "metaphor_phrases", "metaphor_types", "interpretation", "scores"]
    if not isinstance(obj, dict):
        return False, "not_dict"
    for k in need:
        if k not in obj:
            return False, f"missing_{k}"
    if obj["have_metaphor"] not in [0, 1]:
        return False, "bad_have_metaphor"
    if not isinstance(obj["metaphor_phrases"], list):
        return False, "phrases_not_list"
    if not isinstance(obj["metaphor_types"], list):
        return False, "types_not_list"
    if not isinstance(obj["interpretation"], str):
        return False, "interp_not_str"
    if obj["scores"] is not None:
        return False, "scores_must_be_null_in_annotate"

    for t in obj["metaphor_types"]:
        if t not in ALLOWED_TYPES:
            return False, f"bad_type_{t}"

    # check span substring (sau align thì thường pass)
    for sp in obj["metaphor_phrases"]:
        if not isinstance(sp, dict):
            return False, "phrase_item_not_dict"
        if not {"phrase", "start", "end"} <= set(sp.keys()):
            return False, "phrase_item_missing_keys"
        st, ed, ph = sp["start"], sp["end"], sp["phrase"]
        if not isinstance(st, int) or not isinstance(ed, int):
            return False, "start_end_not_int"
        if st < 0 or ed < 0 or st > ed or ed > len(sentence):
            return False, "start_end_out_of_range"
        if sentence[st:ed] != ph:
            return False, "phrase_not_match_substring"

    return True, None


def validate_judge(obj: Any) -> Tuple[bool, Optional[str]]:
    """Validate kết quả judge"""
    if not isinstance(obj, dict):
        return False, "not_dict"
    if "scores" not in obj:
        return False, "missing_scores"
    sc = obj["scores"]
    if sc is None:
        return True, None
    if not isinstance(sc, dict):
        return False, "scores_not_dict"

    for k in SCORE_KEYS:
        if k not in sc:
            return False, f"missing_score_{k}"

    def _is_num(x):
        return isinstance(x, (int, float)) and (not isinstance(x, bool))

    for k in SCORE_KEYS:
        v = sc[k]
        if v is None:
            return False, f"score_{k}_is_null"
        if not _is_num(v):
            return False, f"score_{k}_not_number"
        if v < 0 or v > 4:
            return False, f"score_{k}_out_of_range"

    return True, None
