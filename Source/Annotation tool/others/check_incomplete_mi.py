# check_incomplete_mi.py
import os
import django
from django.db.models import Q

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ViMUNCH.settings")
django.setup()

from task.models import MetaphorInterpretation

def MI_DONE_Q():
    return (
        (Q(interpretation__isnull=False) & ~Q(interpretation__exact=""))
        | ~Q(metaphor_types=[])
        | ~Q(metaphor_phrases=[])
    )

def run():
    total_mi = MetaphorInterpretation.objects.filter(annotator__isnull=False).count()
    complete_mi = MetaphorInterpretation.objects.filter(
        annotator__isnull=False
    ).filter(MI_DONE_Q()).count()

    incomplete_qs = MetaphorInterpretation.objects.filter(
        annotator__isnull=False
    ).exclude(MI_DONE_Q())
    incomplete_count = incomplete_qs.count()

    print("="*50)
    print("📊 Báo cáo kiểm tra MetaphorInterpretation")
    print(f"Tổng MI có annotator: {total_mi}")
    print(f"Số MI được coi là hoàn thành (đủ điều kiện sang PJ): {complete_mi}")
    print(f"Số MI rỗng (không chuyển sang PJ): {incomplete_count}")
    print("="*50)

    print("\n📌 Ví dụ 10 MI rỗng:")
    for mi in incomplete_qs[:10]:
        print(
            f"ID={mi.id}, Dataset={mi.dataset_id}, "
            f"Interpretation='{mi.interpretation}', "
            f"Types={mi.metaphor_types}, Spans={mi.metaphor_phrases}"
        )

if __name__ == "__main__":
    run()
