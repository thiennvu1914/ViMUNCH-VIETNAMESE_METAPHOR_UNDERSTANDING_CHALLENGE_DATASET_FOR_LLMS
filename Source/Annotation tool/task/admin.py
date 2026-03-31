from django.contrib import admin
from django.db.models import OuterRef, Subquery, FloatField
from .models import (
    MetaphorDataset,
    MetaphorInterpretation,
    ParaphraseJudgement,
    TaskAssignment,
)

# ===================== MetaphorDataset =====================
@admin.register(MetaphorDataset)
class MetaphorDatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "short_sentence")
    search_fields = ("sentence",)
    ordering = ("id",)

    def short_sentence(self, obj):
        return (obj.sentence or "")[:80]
    short_sentence.short_description = "Sentence"


# =============== Inline: ParaphraseJudgement ===============
class ParaphraseJudgementInline(admin.TabularInline):
    model = ParaphraseJudgement
    extra = 0
    fields = (
        "annotator",
        "version",
        "paraphrase_sentence",
        "accuracy_score",
        "clarity_score",
        "naturalness_score",
        "meaning_similarity",
        "modality_similarity",
        "implication_similarity",
        "syntax_similarity",
        "context_similarity",
        "overall_similarity",
        "quality_score",
        "notes",
        "created_at",
    )
    readonly_fields = ("overall_similarity", "quality_score", "created_at")
    show_change_link = True


# ===================== MetaphorInterpretation =====================
@admin.register(MetaphorInterpretation)
class MetaphorInterpretationAdmin(admin.ModelAdmin):
    """
    Task 1: diễn giải span + diễn giải câu + phân loại (nhãn câu).
    Hiển thị thêm điểm của bản ParaphraseJudgement mới nhất (Task 2).
    """

    list_display = (
        "id",
        "short_sentence",
        "annotator",
        "spans_count",
        "sentence_types",
        "has_metaphor",
        "latest_overall_similarity",
        "latest_quality_score",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("metaphor_sentence", "annotator__username")
    autocomplete_fields = ("dataset", "annotator")
    ordering = ("id",)
    inlines = [ParaphraseJudgementInline]

    def get_queryset(self, request):
        qs = super().get_queryset(request)

        # Subquery: bản chấm mới nhất theo (version desc, id desc)
        latest_pj = ParaphraseJudgement.objects.filter(
            interpretation=OuterRef("pk")
        ).order_by("-version", "-id")

        return qs.annotate(
            _latest_overall=Subquery(latest_pj.values("overall_similarity")[:1], output_field=FloatField()),
            _latest_quality=Subquery(latest_pj.values("quality_score")[:1], output_field=FloatField()),
        )

    # --- Cột hiển thị ---
    def short_sentence(self, obj):
        return (obj.metaphor_sentence or "")[:80]
    short_sentence.short_description = "Sentence"

    def spans_count(self, obj):
        return len(obj.metaphor_phrases or [])
    spans_count.short_description = "#Spans"

    def sentence_types(self, obj):
        types = obj.metaphor_types or []
        return ", ".join(types[:3]) + (" …" if len(types) > 3 else "")
    sentence_types.short_description = "Sentence Types"

    def has_metaphor(self, obj):
        return bool((obj.metaphor_phrases or []) or (obj.metaphor_types or []))
    has_metaphor.boolean = True
    has_metaphor.short_description = "Has Metaphor"

    def latest_overall_similarity(self, obj):
        return getattr(obj, "_latest_overall", None)
    latest_overall_similarity.short_description = "Overall Sim."

    def latest_quality_score(self, obj):
        return getattr(obj, "_latest_quality", None)
    latest_quality_score.short_description = "Quality"


# ===================== ParaphraseJudgement =====================
@admin.register(ParaphraseJudgement)
class ParaphraseJudgementAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "interpretation",
        "annotator",
        "version",
        "overall_similarity",
        "quality_score",
        "created_at",
    )
    list_filter = ("version", "created_at")
    search_fields = ("paraphrase_sentence", "interpretation__metaphor_sentence", "annotator__username")
    autocomplete_fields = ("interpretation", "annotator")
    ordering = ("interpretation_id", "version", "id")


# ===================== TaskAssignment =====================
@admin.register(TaskAssignment)
class TaskAssignmentAdmin(admin.ModelAdmin):
    list_display = ("annotator", "task_type", "status", "assigned_at", "deadline", "completed_at", "sample_count")
    search_fields = ("annotator__username", "task_type", "status")
    list_filter = ("status", "task_type", "assigned_at", "completed_at")
    autocomplete_fields = ["annotator", "datasets"]   # <-- đã đổi dataset_ids -> datasets
    # Hoặc nếu muốn giao diện chọn nhiều dataset dễ hơn:
    # filter_horizontal = ("datasets",)
    ordering = ("-assigned_at",)
