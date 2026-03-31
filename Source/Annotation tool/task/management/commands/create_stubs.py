import json
from django.core.management.base import BaseCommand
from task.models import MetaphorDataset, MetaphorInterpretation


class Command(BaseCommand):
    help = "Tạo stub (annotator=None) từ file JSON gốc để hiển thị gợi ý"

    def add_arguments(self, parser):
        parser.add_argument("file_path", type=str, help="Đường dẫn đến file JSON gốc")

    def handle(self, *args, **options):
        file_path = options["file_path"]

        self.stdout.write(self.style.NOTICE(f"📂 Đang đọc file: {file_path}"))

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Lỗi khi đọc file JSON: {e}"))
            return

        created_stub = 0
        skipped = 0

        for item in data:
            sentence = item.get("sentence", "").strip()
            phrases = item.get("metaphor_phrases", [])
            types = item.get("metaphor_types", [])

            if not sentence:
                continue

            ds = MetaphorDataset.objects.filter(sentence=sentence).first()
            if not ds:
                skipped += 1
                continue

            exists = MetaphorInterpretation.objects.filter(
                dataset=ds, annotator=None, version=1
            ).exists()

            if not exists:
                MetaphorInterpretation.objects.create(
                    dataset=ds,
                    annotator=None,
                    version=1,
                    metaphor_sentence=sentence,
                    metaphor_phrases=phrases or [],
                    metaphor_types=types or [],
                )
                created_stub += 1
            else:
                skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f"✅ Tạo {created_stub} stub mới, bỏ qua {skipped} mẫu."
        ))
