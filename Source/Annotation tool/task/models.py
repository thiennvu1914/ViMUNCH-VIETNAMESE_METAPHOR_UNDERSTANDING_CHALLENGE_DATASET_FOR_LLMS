from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator

# ======================
# Dataset chứa câu ẩn dụ
# ======================
class MetaphorDataset(models.Model):
    sentence = models.TextField()

    def __str__(self):
        return self.sentence[:50]


# ============================================
# Task 1: Diễn giải từng span + cả câu & phân loại
# ============================================
class MetaphorInterpretation(models.Model):
    annotator = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name="metaphor_interpretations",
        on_delete=models.SET_NULL,
        null=True, blank=True,
    )
    dataset = models.ForeignKey(
        MetaphorDataset, on_delete=models.CASCADE, related_name="interpretations"
    )

    # Câu gốc
    metaphor_sentence = models.TextField()

    # Danh sách cụm ẩn dụ: [{"phrase": "...", "start": int, "end": int, "interpretation": "..."}]
    metaphor_phrases = models.JSONField(default=list, blank=True)

    # Loại ẩn dụ cho toàn câu: lưu list[str] thay cho ArrayField(Postgres-only)
    metaphor_types = models.JSONField(default=list, blank=True)

    # Diễn giải toàn câu
    interpretation = models.TextField(blank=True, default="")

    # Ghi chú
    notes = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    version = models.IntegerField(default=1, validators=[MinValueValidator(1)])

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["annotator", "dataset", "version"],
                name="uq_mi_annotator_dataset_version",
            ),
        ]
        indexes = [
            models.Index(fields=["annotator", "dataset"]),
        ]

    def __str__(self):
        return f"[MI] {self.metaphor_sentence[:50]}"


# ======================================
# Task 2: Đánh giá chất lượng diễn giải
# ======================================
class ParaphraseJudgement(models.Model):
    interpretation = models.ForeignKey(
        MetaphorInterpretation, on_delete=models.CASCADE, related_name="judgements"
    )

    # Khuyến nghị: đặt null=False khi kết thúc giai đoạn chuyển đổi
    annotator = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="paraphrase_judgements",
        null=True,    # tạm thời cho phép null để migrate/điền dữ liệu
        blank=True,
    )

    paraphrase_sentence = models.TextField(blank=True, default="")

    # Điểm đánh giá (ví dụ 1..5)
    accuracy_score = models.IntegerField(null=True, blank=True,
                                         validators=[MinValueValidator(1), MaxValueValidator(5)])
    clarity_score = models.IntegerField(null=True, blank=True,
                                        validators=[MinValueValidator(1), MaxValueValidator(5)])
    naturalness_score = models.IntegerField(null=True, blank=True,
                                            validators=[MinValueValidator(1), MaxValueValidator(5)])

    meaning_similarity = models.IntegerField(null=True, blank=True,
                                             validators=[MinValueValidator(0), MaxValueValidator(100)])
    modality_similarity = models.IntegerField(null=True, blank=True,
                                              validators=[MinValueValidator(0), MaxValueValidator(100)])
    implication_similarity = models.IntegerField(null=True, blank=True,
                                                 validators=[MinValueValidator(0), MaxValueValidator(100)])
    syntax_similarity = models.IntegerField(null=True, blank=True,
                                            validators=[MinValueValidator(0), MaxValueValidator(100)])
    context_similarity = models.IntegerField(null=True, blank=True,
                                             validators=[MinValueValidator(0), MaxValueValidator(100)])

    # Nên dùng Decimal cho điểm tổng hợp
    overall_similarity = models.DecimalField(null=True, blank=True, max_digits=5, decimal_places=2)
    quality_score = models.DecimalField(null=True, blank=True, max_digits=5, decimal_places=2)

    notes = models.TextField(blank=True, default="")

    version = models.IntegerField(default=1, validators=[MinValueValidator(1)])
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            # Khi đã kết thúc giai đoạn null annotator, set annotator null=False rồi giữ unique này
            models.UniqueConstraint(
                fields=["interpretation", "annotator", "version"],
                name="uq_pj_interpretation_annotator_version",
            ),
        ]
        indexes = [
            models.Index(fields=["interpretation", "version"]),
        ]

    def save(self, *args, **kwargs):
        # Tính toán độ tương đồng tổng thể (trọng số: 30-25-25-10-10; thang 0..100)
        if all(v is not None for v in [
            self.meaning_similarity,
            self.modality_similarity,
            self.implication_similarity,
            self.syntax_similarity,
            self.context_similarity
        ]):
            self.overall_similarity = (
                (self.meaning_similarity * 30 +
                 self.modality_similarity * 25 +
                 self.implication_similarity * 25 +
                 self.syntax_similarity * 10 +
                 self.context_similarity * 10) / 100
            )

        # Tính toán chất lượng diễn giải (giả sử 1..5)
        if all(v is not None for v in [
            self.accuracy_score,
            self.clarity_score,
            self.naturalness_score
        ]):
            self.quality_score = (
                (self.accuracy_score * 3 +
                 self.clarity_score * 2 +
                 self.naturalness_score) / 6
            )

        super().save(*args, **kwargs)

    def __str__(self):
        return f"[PJ] {self.interpretation.metaphor_sentence[:50]} - v{self.version}"


# ======================================
# Gán task cho annotator
# ======================================
class TaskAssignment(models.Model):
    annotator = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="task_assignments"
    )
    task_type = models.CharField(max_length=50)

    # Đặt tên rõ nghĩa hơn
    datasets = models.ManyToManyField(MetaphorDataset)

    status = models.CharField(
        max_length=50,
        choices=[('assigned', 'Assigned'), ('completed', 'Completed')]
    )
    deadline = models.DateTimeField()
    assigned_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    sample_count = models.IntegerField(default=1, validators=[MinValueValidator(1)])

    def __str__(self):
        return f"Task assigned to {self.annotator.username} with {self.sample_count} samples"
