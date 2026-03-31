# forms.py
import json
from django import forms
from django.db.models import Q
from .models import (
    MetaphorDataset,
    MetaphorInterpretation,
    ParaphraseJudgement,
    TaskAssignment,
)
from users.models import User

# ======= Mapping & Choices (GIÁ TRỊ LÀ MÃ VỮNG - labels là TV) =======
# Thêm 'khác' → 'other' để đồng bộ
TYPE_VALUE_MAP = {
    "ẩn dụ cảm xúc": "emotional",
    "ẩn dụ cấu trúc": "structural",
    "ẩn dụ bản thể": "ontological",
    "ẩn dụ định hướng": "orientational",
    "ẩn dụ văn hóa dân gian": "cultural_folklore",
    "ẩn dụ văn hoá dân gian": "cultural_folklore",
    "cultural": "cultural_folklore",
    "khác": "other",
}
# Choices hiển thị cho checkbox (value là mã, label là TV)
SENTENCE_TYPE_CHOICES = [
    ("emotional", "Ẩn dụ cảm xúc"),
    ("structural", "Ẩn dụ cấu trúc"),
    ("ontological", "Ẩn dụ bản thể"),
    ("orientational", "Ẩn dụ định hướng"),
    ("cultural_folklore", "Ẩn dụ văn hóa dân gian"),
    ("other", "Khác"),
]

def _normalize_type_value(x: str):
    if not x:
        return ""
    s = str(x).strip()
    low = s.lower()
    # nếu đã là mã vững thì giữ nguyên
    if low in {v for v in TYPE_VALUE_MAP.values()}:
        return low
    # nếu là nhãn/alias → quy về mã
    return TYPE_VALUE_MAP.get(low, low)


class MetaphorDatasetForm(forms.ModelForm):
    class Meta:
        model = MetaphorDataset
        fields = ["sentence"]


# ===================== TASK 1 =====================
class MetaphorInterpretationForm(forms.ModelForm):
    # Span-level (JSON [{phrase,start,end,interpretation,type}])
    metaphor_phrases = forms.CharField(widget=forms.HiddenInput(), required=False)

    # Sentence-level multi-label cho CẢ CÂU → lưu vào JSONField metaphor_types (MÃ VỮNG)
    metaphor_types = forms.MultipleChoiceField(
        choices=SENTENCE_TYPE_CHOICES,
        required=False,
        widget=forms.CheckboxSelectMultiple,
        label="Phân loại ẩn dụ cho CẢ CÂU (chọn nhiều)",
    )

    class Meta:
        model = MetaphorInterpretation
        fields = [
            "metaphor_types",
            "interpretation",
            "notes",
            "metaphor_phrases",
        ]
        labels = {
            "interpretation": "Diễn giải cả câu",
            "notes": "Ghi chú (nếu có)",
        }
        widgets = {
            "interpretation": forms.Textarea(attrs={"rows": 4, "class": "form-control"}),
            "notes": forms.Textarea(attrs={"rows": 2, "class": "form-control"}),
        }

    def __init__(self, *args, **kwargs):
        # Prefill span JSON để highlight lúc load trang
        inst = kwargs.get("instance")
        if inst and isinstance(inst.metaphor_phrases, list):
            initial = kwargs.setdefault("initial", {})
            initial["metaphor_phrases"] = json.dumps(inst.metaphor_phrases, ensure_ascii=False)

        super().__init__(*args, **kwargs)

        # Prefill nhãn câu: luôn chuẩn hoá về mã vững để checkbox tick chuẩn
        current_initial = self.initial.get("metaphor_types")
        if current_initial is None and inst and isinstance(inst.metaphor_types, list):
            current_initial = inst.metaphor_types

        if current_initial is not None:
            norm = [_normalize_type_value(x) for x in (current_initial or [])]
            # unique + giữ thứ tự
            seen, uniq = set(), []
            for t in norm:
                if t and t not in seen:
                    seen.add(t); uniq.append(t)
            self.fields["metaphor_types"].initial = uniq

    def clean_metaphor_types(self):
        data = self.cleaned_data.get("metaphor_types") or []
        # map mọi thứ về mã vững (phòng khi POST lên là nhãn)
        norm = [_normalize_type_value(x) for x in data]
        # unique + giữ thứ tự
        seen, uniq = set(), []
        for t in norm:
            if t and t not in seen:
                seen.add(t); uniq.append(t)
        return uniq

    def clean_metaphor_phrases(self):
        data = self.cleaned_data.get("metaphor_phrases", "[]")
        try:
            spans = json.loads(data) if data else []
            # Validate (thêm default để không lỗi nếu thiếu key từ import cũ)
            for i, span in enumerate(spans):
                if not isinstance(span, dict):
                    raise forms.ValidationError("Dữ liệu span không hợp lệ.")
                for k in ("phrase", "start", "end"):
                    if k not in span:
                        raise forms.ValidationError("Thiếu thông tin span.")
                if not isinstance(span["start"], int) or not isinstance(span["end"], int):
                    raise forms.ValidationError("Vị trí bắt đầu/kết thúc không hợp lệ.")
                # mặc định nếu thiếu
                span.setdefault("interpretation", "")
                span.setdefault("type", "")
            return spans
        except Exception:
            raise forms.ValidationError("Dữ liệu các span không hợp lệ!")


# ===================== TASK 2 =====================
class ParaphraseJudgementForm(forms.ModelForm):
    class Meta:
        model = ParaphraseJudgement
        fields = [
            "meaning_similarity",
            "modality_similarity",
            "implication_similarity",
            "syntax_similarity",
            "context_similarity",
            "accuracy_score",
            "clarity_score",
            "naturalness_score",
            "notes",
        ]
        labels = {
            "meaning_similarity": "Tương đồng về nghĩa mệnh đề",
            "modality_similarity": "Tương đồng về nghĩa tình thái",
            "implication_similarity": "Tương đồng về hàm ý",
            "syntax_similarity": "Tương đồng về cấu trúc ngữ pháp",
            "context_similarity": "Tương đồng về ngữ cảnh sử dụng",
            "accuracy_score": "Độ chính xác từ MI",
            "clarity_score": "Độ rõ ràng từ MI",
            "naturalness_score": "Độ tự nhiên từ MI",
            "notes": "Ghi chú (nếu có)",
        }
        # NOTE: nếu model cho phép 0..100 thì đổi choices tương ứng.
        widgets = {
            field: forms.Select(choices=[(i, i) for i in range(5)], attrs={"class": "form-select"})
            for field in [
                "meaning_similarity",
                "modality_similarity",
                "implication_similarity",
                "syntax_similarity",
                "context_similarity",
                "accuracy_score",
                "clarity_score",
                "naturalness_score",
            ]
        }
        widgets["notes"] = forms.Textarea(attrs={"class": "form-control", "rows": 3})


# ===================== ASSIGN =====================
class TaskAssignmentForm(forms.ModelForm):
    annotator = forms.ModelChoiceField(
        queryset=User.objects.filter(Q(role="annotator") | Q(role="senior_annotator")),
        label="Người gán nhãn",
    )
    sample_count = forms.IntegerField(min_value=1, label="Số lượng mẫu cần gán")
    deadline = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}), label="Ngày deadline")
    task_type = forms.ChoiceField(
        choices=[
            ("Metaphor Interpretation & Classification", "Metaphor Interpretation & Classification"),
            ("Paraphrase Judgement", "Paraphrase Judgement"),
        ],
        widget=forms.HiddenInput(),
    )

    class Meta:
        model = TaskAssignment
        fields = ["annotator", "task_type", "deadline", "sample_count"]
