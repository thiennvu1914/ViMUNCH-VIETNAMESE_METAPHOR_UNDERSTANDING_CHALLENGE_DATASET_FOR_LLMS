from django.core.management.base import BaseCommand
from task.models import MetaphorInterpretation

class Command(BaseCommand):
    help = "Xoá tất cả stub MetaphorInterpretation (annotator=None) và in báo cáo trước/sau."

    def handle(self, *args, **options):
        total_before = MetaphorInterpretation.objects.count()
        stub_qs = MetaphorInterpretation.objects.filter(annotator__isnull=True)
        stub_count = stub_qs.count()

        self.stdout.write(self.style.WARNING(f"Tổng số record trước khi xoá: {total_before}"))
        self.stdout.write(self.style.WARNING(f"Trong đó có {stub_count} stub (annotator=None)"))

        if stub_count == 0:
            self.stdout.write(self.style.SUCCESS("Không có stub nào để xoá."))
            return

        # Tiến hành xoá
        deleted, _ = stub_qs.delete()

        total_after = MetaphorInterpretation.objects.count()
        self.stdout.write(self.style.SUCCESS(f"Đã xoá {deleted} stub."))
        self.stdout.write(self.style.SUCCESS(f"Tổng số record sau khi xoá: {total_after}"))
