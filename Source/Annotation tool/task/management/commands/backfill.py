from django.core.management.base import BaseCommand

print("🔥 backfill.py được import!")

class Command(BaseCommand):
    help = "Test backfill command"

    def handle(self, *args, **options):
        print("✅ Backfill command chạy được!")
