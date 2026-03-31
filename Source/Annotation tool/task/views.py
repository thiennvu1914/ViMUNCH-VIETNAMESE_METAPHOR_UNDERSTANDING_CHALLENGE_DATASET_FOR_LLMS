import json, ast
import io
import datetime
from django.utils import timezone
from django.core.serializers.json import DjangoJSONEncoder

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Exists, OuterRef, Q, Max, Prefetch
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView
from django.db import transaction

from users.models import User
from .models import (
    MetaphorDataset,
    MetaphorInterpretation,
    ParaphraseJudgement,
    TaskAssignment,
)
from .forms import (
    MetaphorDatasetForm,
    TaskAssignmentForm,
    MetaphorInterpretationForm,
    ParaphraseJudgementForm,
)

# ---------- Helpers ----------
def is_admin(user):
    return user.is_authenticated and getattr(user, "role", None) == "admin"

def admin_required(view_func):
    def _wrapped_view(request, *args, **kwargs):
        if not is_admin(request.user):
            return redirect("task:dashboard")
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def MI_DONE_Q():
    return (
        (Q(interpretation__isnull=False) & ~Q(interpretation__exact=""))
        | ~Q(metaphor_types=[])
        | ~Q(metaphor_phrases=[])
    )
    
# ===================== Admin: Quản lý task =====================
@login_required
@admin_required
def manage_task(request):
    # MẪU SỐ: TOÀN BỘ DATASET
    total_dataset = MetaphorDataset.objects.count()

    # MI done: chỉ tính bản của NGƯỜI THẬT (loại stub annotator=None)
    t1_done_ds_ids = (
        MetaphorInterpretation.objects
        .filter(annotator__isnull=False)
        .values_list("dataset_id", flat=True)
        .distinct()
    )
    total_interpret_classification = len(set(t1_done_ds_ids))
    pct_ic = round((total_interpret_classification / total_dataset) * 100, 2) if total_dataset else 0.0

    # PJ: mẫu số là số MI đủ điều kiện (MI done), tử số là số MI đã có ít nhất 1 judgement của NGƯỜI THẬT
    eligible_interp_ids = (
        MetaphorInterpretation.objects
        .filter(annotator__isnull=False)
        .filter(MI_DONE_Q())
        .values_list("id", flat=True)
        .distinct()
    )
    total_interpret = len(set(eligible_interp_ids))
    total_judgement = (
        ParaphraseJudgement.objects
        .filter(annotator__isnull=False, interpretation_id__in=eligible_interp_ids)
        .values("interpretation_id").distinct().count()
    )
    pct_pj = round((total_judgement / total_interpret) * 100, 2) if total_interpret else 0.0

    tasks = (
        TaskAssignment.objects
        .select_related("annotator")
        .prefetch_related("datasets")
        .order_by("-assigned_at")
    )

    context = {
        "tasks": tasks,
        "total_dataset": total_dataset,                          # 22
        "total_interpret_classification": total_interpret_classification,  # 4
        "pct_ic": pct_ic,
        "total_interpret": total_interpret,
        "total_judgement": total_judgement,
        "pct_pj": pct_pj,
        "progress_percent": pct_ic,
    }
    return render(request, "task/manage_task.html", context)

# ===================== Annotator dashboard (2 task) =====================
@login_required
def annotator_dashboard(request):
    user = request.user
    if is_admin(user):
        return redirect("task:manage_task")  # admin không xem dashboard annotator

    mi_assign_qs = (
        TaskAssignment.objects
        .filter(annotator=user, task_type="Metaphor Interpretation & Classification")
        .prefetch_related("datasets")
    )
    pj_assign_qs = (
        TaskAssignment.objects
        .filter(annotator=user, task_type="Paraphrase Judgement")
        .prefetch_related("datasets")
    )

    assigned_interpret  = mi_assign_qs.values("datasets__id").distinct().count()
    assigned_judgement  = pj_assign_qs.values("datasets__id").distinct().count()
    completed_interpret = (
        MetaphorInterpretation.objects
        .filter(annotator=user)
        .values("dataset_id")
        .distinct()
        .count()
    )
    completed_judgement = (
        ParaphraseJudgement.objects.filter(annotator=user)
        .values("interpretation_id").distinct().count()
    )

    context = {
        "assigned_interpret": assigned_interpret,
        "completed_interpret": completed_interpret,
        "assigned_judgement": assigned_judgement,
        "completed_judgement": completed_judgement,
    }
    return render(request, "task/annotator_dashboard.html", context)

# ===================== Assign task =====================

# ====== tên task chuẩn hoá ======
MI_TASK = 'Metaphor Interpretation & Classification'
PJ_TASK = 'Paraphrase Judgement'

# ====== helpers: tập id dataset theo các tiêu chí ======
def _assigned_ids_anyone(task_type: str) -> set[int]:
    """Tất cả dataset đã được GIAO cho BẤT KỲ annotator nào ở task_type."""
    return set(
        TaskAssignment.objects
        .filter(task_type=task_type)
        .values_list('datasets__id', flat=True)
    )

def _assigned_ids_of_user(task_type: str, user) -> set[int]:
    """Tất cả dataset đã được GIAO cho annotator này ở task_type."""
    return set(
        TaskAssignment.objects
        .filter(task_type=task_type, annotator=user)
        .values_list('datasets__id', flat=True)
    )

def _mi_done_ids_anyone() -> set[int]:
    """Dataset đã HOÀN THÀNH MI (bởi ai cũng được)."""
    return set(
        MetaphorInterpretation.objects
        .filter(annotator__isnull=False)  # bỏ stub
        .filter(
            Q(interpretation__isnull=False) & ~Q(interpretation__exact="") |
            ~Q(metaphor_types=[]) |
            ~Q(metaphor_phrases=[])
        )
        .values_list('dataset_id', flat=True)
    )

def _mi_done_ids_of_user(user) -> set[int]:
    """Dataset đã được annotator này làm MI (để không giao lại cho chính họ)."""
    return set(
        MetaphorInterpretation.objects
        .filter(annotator=user)
        .values_list('dataset_id', flat=True)
    )

def _pj_done_ids_of_user(user) -> set[int]:
    """Dataset đã được annotator này chấm PJ (để không giao lại cho chính họ)."""
    return set(
        ParaphraseJudgement.objects
        .filter(annotator=user)
        .values_list('interpretation__dataset_id', flat=True)
    )

def _mi_ready_ids_anyone() -> set[int]:
    """Dataset đã có interpretation (đủ điều kiện đưa sang PJ)."""
    return set(
        MetaphorInterpretation.objects
        .filter(Q(interpretation__isnull=False) & ~Q(interpretation__exact=""))
        .values_list('dataset_id', flat=True)
        .distinct()
    )

# ============== Giao task ==============
@login_required
@admin_required
def assign_task(request, task_type):
    from datetime import datetime, time
    from django.utils import timezone

    if request.method == 'POST':
        form = TaskAssignmentForm(request.POST)
        if form.is_valid():
            annotator   = form.cleaned_data['annotator']
            num_samples = form.cleaned_data['sample_count']
            deadline    = form.cleaned_data['deadline']  # DateField

            # ---- các tập id dùng để loại trừ ----
            # (1) đã làm bởi chính annotator này
            if task_type == MI_TASK:
                done_by_user = _mi_done_ids_of_user(annotator)
            elif task_type == PJ_TASK:
                done_by_user = _pj_done_ids_of_user(annotator)
            else:
                messages.error(request, f'Loại task không hợp lệ: {task_type}')
                return redirect('task:assign_task', task_type=task_type)

            # (2) đã từng được giao cho chính annotator này ở task đang xét
            already_assigned_to_user = _assigned_ids_of_user(task_type, annotator)

            # (3) đã được giao cho NGƯỜI KHÁC ở task đang xét (tránh trùng người)
            already_assigned_anyone = _assigned_ids_anyone(task_type)

            if task_type == MI_TASK:
                # (4) MI đã hoàn thành bởi ai đó -> không giao lại
                mi_done_anyone = _mi_done_ids_anyone()

                exclude_ids = done_by_user | already_assigned_to_user | \
                              already_assigned_anyone | mi_done_anyone

                pool_qs = MetaphorDataset.objects.exclude(id__in=exclude_ids).order_by('id')

            else:  # PJ_TASK
                # chỉ lấy những câu đã có interpretation
                mi_ready_ids = _mi_ready_ids_anyone()

                # cross-check: KHÔNG giao PJ những câu mà annotator này đã được GIAO ở MI
                mi_assigned_to_user = _assigned_ids_of_user(MI_TASK, annotator)

                exclude_ids = done_by_user | already_assigned_to_user | \
                              already_assigned_anyone | mi_assigned_to_user

                candidate_ids = list(set(mi_ready_ids) - exclude_ids)
                pool_qs = MetaphorDataset.objects.filter(id__in=candidate_ids).order_by('id')

            # lấy mẫu
            pick = list(pool_qs.values_list('id', flat=True)[:num_samples])
            if not pick:
                messages.error(request, f'Không còn mẫu phù hợp cho {task_type}.')
                return redirect('task:assign_task', task_type=task_type)

            # tạo/cập nhật assignment (gom theo annotator + task_type)
            deadline_dt = timezone.make_aware(datetime.combine(deadline, time.min))
            ta, _ = TaskAssignment.objects.get_or_create(
                annotator=annotator,
                task_type=task_type,
                defaults={'deadline': deadline_dt, 'status': 'assigned'}
            )
            ta.datasets.add(*pick)
            ta.deadline = deadline_dt
            ta.sample_count = ta.datasets.count()
            ta.save()

            messages.success(
                request, f'Đã gán {len(pick)} mẫu cho {annotator.username} ({task_type}).'
            )
            return redirect('task:assign_task', task_type=task_type)

        messages.error(request, 'Thông tin không hợp lệ. Vui lòng kiểm tra lại.')
    else:
        form = TaskAssignmentForm(initial={'task_type': task_type})

    users = User.objects.filter(role__in=['annotator', 'senior_annotator'])
    return render(request, 'task/assign_task.html', {
        'form': form,
        'users': users,
        'selected_task_type': task_type
    })


# ===================== Danh sách assignment (admin) =====================
def update_task_completion(annotator, task_type):
    """
    Cập nhật trạng thái TaskAssignment (status, completed_at)
    khi annotator đã submit toàn bộ dataset được giao.
    Hợp lệ cả khi câu không có ẩn dụ (vẫn có bản ghi MI).
    """
    tasks = TaskAssignment.objects.filter(annotator=annotator, task_type=task_type)

    for task in tasks:
        assigned_ids = set(task.datasets.values_list("id", flat=True))
        if not assigned_ids:
            continue

        # --- Lấy danh sách ID dataset mà annotator đã làm ---
        if task_type == "Metaphor Interpretation & Classification":
            # ✅ Coi là đã làm nếu có bất kỳ bản ghi MI hợp lệ nào cho dataset đó
            done_ids = set(
                MetaphorInterpretation.objects
                .filter(annotator=annotator, dataset_id__in=assigned_ids)
                .filter(
                    Q(interpretation__isnull=False)
                    | ~Q(metaphor_phrases=[])
                    | ~Q(metaphor_types=[])
                )
                .values_list("dataset_id", flat=True)
            )
        elif task_type == "Paraphrase Judgement":
            done_ids = set(
                ParaphraseJudgement.objects
                .filter(annotator=annotator, interpretation__dataset_id__in=assigned_ids)
                .values_list("interpretation__dataset_id", flat=True)
            )
        else:
            continue

        # --- Cập nhật trạng thái ---
        if assigned_ids.issubset(done_ids):
            if task.status != "completed":
                task.status = "completed"
                task.completed_at = timezone.now()
                task.save(update_fields=["status", "completed_at"])
        else:
            if task.status == "completed":
                task.status = "assigned"
                task.completed_at = None
                task.save(update_fields=["status", "completed_at"])

            
@login_required
def task_assignment_list(request):
    assignments = (
        TaskAssignment.objects
        .select_related('annotator')
        .prefetch_related('datasets')
        .order_by('deadline')
    )

    # === NEW: thêm thông tin tiến độ ===
    data = []
    for a in assignments:
        total = a.datasets.count()

        if a.task_type == "Metaphor Interpretation & Classification":
            done = (
                MetaphorInterpretation.objects
                .filter(
                    annotator=a.annotator,
                    dataset_id__in=a.datasets.values_list("id", flat=True)
                )
                .filter(
                    Q(interpretation__isnull=False)
                    | ~Q(metaphor_phrases=[])
                    | ~Q(metaphor_types=[])
                )
                .values("dataset_id").distinct().count()
            )
        elif a.task_type == "Paraphrase Judgement":
            done = (
                ParaphraseJudgement.objects
                .filter(
                    annotator=a.annotator,
                    interpretation__dataset_id__in=a.datasets.values_list("id", flat=True)
                )
                .values("interpretation__dataset_id").distinct().count()
            )
        else:
            done = 0

        percent = round(done / total * 100, 1) if total > 0 else 0

        data.append({
            "assignment": a,
            "done": done,
            "total": total,
            "percent": percent,
        })

    return render(request, "task/task_assignment_list.html", {"assignments": data})

# ===================== Dashboard (admin) =====================
@login_required
@user_passes_test(is_admin)
def dashboard_view(request):

    total_dataset = MetaphorDataset.objects.count()

    t1_done = (
        MetaphorInterpretation.objects
        .filter(
            Q(interpretation__isnull=False) & ~Q(interpretation__exact="") |
            ~Q(metaphor_types=[]) |
            ~Q(metaphor_phrases=[])
        )
        .values("dataset_id").distinct().count()
    )

    t2_done = ParaphraseJudgement.objects.values("interpretation_id").distinct().count()

    annotator_progress = get_annotator_progress()

    context = {
        "interpretation_done": t1_done,
        "judgement_done": t2_done,
        "total_dataset": total_dataset,
        "annotator_progress": annotator_progress,
    }
    return render(request, "task/dashboard.html", context)

def get_annotator_progress():
    annotators = User.objects.filter(role="annotator")
    rows = []
    total_dataset = MetaphorDataset.objects.count()

    for a in annotators:
        # Task 1 của annotator a
        t1_done = (
            MetaphorInterpretation.objects.filter(annotator=a)
            .filter(
                Q(interpretation__isnull=False) & ~Q(interpretation__exact="") |
                ~Q(metaphor_types=[]) |
                ~Q(metaphor_phrases=[])
            )
            .count()
        )
        t1_total = (
            TaskAssignment.objects
            .filter(annotator=a, task_type="Metaphor Interpretation & Classification")
            .values("datasets__id").distinct().count()   # <-- FIX
        )

        # Task 2 do a chấm
        t2_done = ParaphraseJudgement.objects.filter(annotator=a).count()
        t2_total = (
            TaskAssignment.objects
            .filter(annotator=a, task_type="Paraphrase Judgement")
            .values("datasets__id").distinct().count()   # <-- FIX
        )

        rows.append({
            "annotator": a,
            "interpretation": f"{t1_done}/{t1_total or total_dataset}",
            "judgement": f"{t2_done}/{t2_total or t1_done}",
        })
    return rows

# ===================== Admin: Import CSV đơn giản =====================

# ====== Map loại ẩn dụ (đồng bộ FE/BE) ======
TYPE_MAP_VI_EN = {
    "ẩn dụ cấu trúc": "structural",
    "ẩn dụ bản thể": "ontological",
    "ẩn dụ cảm xúc": "emotional",
    "ẩn dụ định hướng": "orientational",
    "ẩn dụ văn hóa dân gian": "cultural_folklore",
    "cultural": "cultural_folklore",
    "khác": "other",
}

KEY_ALIASES = {
    "sentence": ["sentence", "text", "content", "raw", "input"],
    "interpretation": ["interpretation", "paraphrase", "explanation"],
    "metaphor_phrases": ["metaphor_phrases", "spans", "phrases", "metaphor_spans"],
    "metaphor_types": ["metaphor_types", "types", "labels", "met_types"],
    "source": ["source", "src"],
}

def _pick(obj, field):
    for k in KEY_ALIASES.get(field, []):
        if k in obj:
            return obj[k]
    return None

def _read_json_or_jsonl_from_bytes(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="ignore") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    items = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        try:
            items.append(json.loads(t))
        except Exception:
            items.append(t)
    return items

def _to_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["data", "items", "records", "rows"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        return [payload]
    return []

def _normalize_phrases(value, sentence: str | None = None):
    """
    Chuẩn hoá về list[dict]: {phrase, start, end}
    Hỗ trợ: dict | [text,start,end] | "text" | chuỗi JSON của các dạng trên.
    Tự sửa 'end-inclusive' (+1) & trim khoảng trắng ở rìa.
    """
    def _find_next(haystack: str, needle: str, used_ranges):
        if not haystack or not needle:
            return -1
        start = 0
        while True:
            idx = haystack.find(needle, start)
            if idx < 0:
                return -1
            rng = (idx, idx + len(needle))
            if all(not (max(rng[0], u[0]) < min(rng[1], u[1])) for u in used_ranges):
                used_ranges.append(rng)
                return idx
            start = idx + 1

    def _fix_indices(st, en, phrase):
        if not (isinstance(st, int) and isinstance(en, int) and st < en and sentence):
            return st, en
        # trim space hai đầu theo câu thực
        while st < en and sentence[st] == " ":
            st += 1
        while st < en and sentence[en - 1] == " ":
            en -= 1
        # nếu không khớp phrase, thử coi end là inclusive (+1)
        if phrase and sentence[st:en] != phrase:
            if en <= len(sentence) - 1 and sentence[st:en + 1] == phrase:
                en = en + 1
            else:
                # fallback tìm theo phrase
                idx = _find_next(sentence, phrase, used)
                if idx != -1:
                    return idx, idx + len(phrase)
        return st, en

    # parse nếu là chuỗi
    if isinstance(value, str):
        v = value.strip()
        try:
            value = json.loads(v)
        except Exception:
            try:
                value = ast.literal_eval(v)
            except Exception:
                value = [v]

    if not isinstance(value, list):
        value = []

    out, used = [], []

    for it in value:
        # object
        if isinstance(it, dict):
            phrase = (it.get("phrase") or it.get("text") or "").strip()
            st = it.get("start"); en = it.get("end")
            if isinstance(st, int) and isinstance(en, int) and st < en:
                st, en = _fix_indices(st, en, phrase)
                if isinstance(st, int) and isinstance(en, int) and st < en:
                    out.append({"phrase": phrase, "start": st, "end": en})
            elif sentence and phrase:
                idx = _find_next(sentence, phrase, used)
                if idx != -1:
                    out.append({"phrase": phrase, "start": idx, "end": idx + len(phrase)})
            continue

        # list [text, start, end, ...]
        if isinstance(it, list) and len(it) >= 3:
            phrase = str(it[0]).strip()
            try:
                st, en = int(it[1]), int(it[2])
            except Exception:
                st = en = None
            if isinstance(st, int) and isinstance(en, int) and st < en:
                st, en = _fix_indices(st, en, phrase)
                if isinstance(st, int) and isinstance(en, int) and st < en:
                    out.append({"phrase": phrase, "start": st, "end": en})
            elif sentence and phrase:
                idx = _find_next(sentence, phrase, used)
                if idx != -1:
                    out.append({"phrase": phrase, "start": idx, "end": idx + len(phrase)})
            continue

        # chỉ chuỗi "phrase"
        if isinstance(it, str):
            phrase = it.strip()
            if sentence and phrase:
                idx = _find_next(sentence, phrase, used)
                if idx != -1:
                    out.append({"phrase": phrase, "start": idx, "end": idx + len(phrase)})

    # unique theo (start,end)
    seen, uniq = set(), []
    for d in out:
        key = (d["start"], d["end"])
        if key not in seen:
            seen.add(key); uniq.append(d)
    return uniq

def _normalize_types(value):
    if isinstance(value, str):
        v = value.strip()
        try:
            value = json.loads(v)
        except Exception:
            try:
                value = ast.literal_eval(v)
            except Exception:
                value = [v]
    if not isinstance(value, list):
        value = [value] if value else []
    out = []
    for it in value:
        if not isinstance(it, str):
            continue
        s = it.strip().strip('"').strip("'")
        if len(s) > 60:
            continue
        k = s.lower()
        out.append(TYPE_MAP_VI_EN.get(k, k))
    seen, uniq = set(), []
    for t in out:
        if t and t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

def _import_dataset_json_and_init_mi_from_bytes(raw_bytes):
    payload = _read_json_or_jsonl_from_bytes(raw_bytes)
    items = _to_items(payload)
    total = len(items) if isinstance(items, list) else 0
    if not total:
        return 0, 0, 0, 0, None

    created = updated = skipped = 0
    preview = items[0] if items else None

    with transaction.atomic():
        for it in items:
            if isinstance(it, str):
                sentence = it.strip()
                source, interp, phrases, types = "", "", [], []
            elif isinstance(it, dict):
                sentence = (_pick(it, "sentence") or "").strip()
                if not sentence and len(it) == 1:
                    sentence = str(list(it.values())[0] or "").strip()
                interp  = (_pick(it, "interpretation") or "").strip()
                phrases = _normalize_phrases(_pick(it, "metaphor_phrases"), sentence)
                types   = _normalize_types(_pick(it, "metaphor_types"))
                source  = (_pick(it, "source") or "").strip()
            else:
                skipped += 1
                continue

            if not sentence:
                skipped += 1
                continue

            ds_defaults = {}
            if hasattr(MetaphorDataset, "source") and source:
                ds_defaults["source"] = source

            ds, ds_created = MetaphorDataset.objects.get_or_create(
                sentence=sentence, defaults=ds_defaults
            )
            if ds_created:
                created += 1
            else:
                # update thêm source nếu có
                if hasattr(MetaphorDataset, "source") and source and not getattr(ds, "source", None):
                    ds.source = source
                    ds.save(update_fields=["source"])
                    updated += 1
                else:
                    skipped += 1

    return created, updated, skipped, total, preview



@login_required
@admin_required
def import_data(request):
    if request.method == "POST":
        upfile = request.FILES.get("json_file")
        if not upfile:
            messages.error(request, "Vui lòng chọn file .json")
            return redirect("task:import_data")
        if not upfile.name.lower().endswith(".json"):
            messages.error(request, "Chỉ hỗ trợ file .json")
            return redirect("task:import_data")

        raw = upfile.read()
        try:
            created, updated, skipped, total, preview = _import_dataset_json_and_init_mi_from_bytes(raw)
            print(f"[IMPORT] file={upfile.name} total={total} created={created} updated={updated} skipped={skipped}")
            if preview is not None:
                print(f"[IMPORT] first_item_preview={str(preview)[:200]}")
            if total == 0:
                messages.error(request, "File JSON rỗng hoặc không đúng định dạng (không tìm thấy mảng records).")
            else:
                messages.success(request, f"Import xong: tổng {total}, tạo {created}, cập nhật {updated}, bỏ qua {skipped}.")
        except Exception as e:
            import traceback
            print("[IMPORT][ERROR]\n", traceback.format_exc())
            messages.error(request, f"Lỗi khi import: {e}")
        return redirect("task:manage_task")

    return render(request, "task/import_data.html")

# ===================== Export CSV (Task 1 + Task 2) =====================

@login_required
@admin_required     
def export_classification(request):
    """
    Xuất JSON tối giản cho mỗi MI:
      sentence, interpretation, metaphor_types, metaphor_phrases[{phrase,start,end,interpretation}], notes, scores (latest PJ)
    Chỉ xuất MI của người thật và đã thực sự làm.
    """

    # Prefetch toàn bộ PJ của người thật; lấy phần tử cuối làm latest
    pj_qs = (
        ParaphraseJudgement.objects
        .filter(annotator__isnull=False)
        .order_by("version", "id")
    )

    mi_qs = (
        MetaphorInterpretation.objects
        .select_related("dataset", "annotator")
        .prefetch_related(Prefetch("judgements", queryset=pj_qs, to_attr="pj_list"))
        .filter(annotator__isnull=False)  # loại stub
        .filter(MI_DONE_Q())              # chỉ MI đã làm
        .order_by("dataset_id", "id")
    )

    out = []
    for mi in mi_qs:
        # Rút gọn phrases
        slim_phrases = []
        for s in (mi.metaphor_phrases or []):
            if isinstance(s, dict):
                slim_phrases.append({
                    "phrase": s.get("phrase") or s.get("text") or "",
                    "start":  int(s.get("start") or 0),
                    "end":    int(s.get("end") or 0),
                    "interpretation": s.get("interpretation") or "",
                })
            elif isinstance(s, (list, tuple)) and len(s) >= 3:
                slim_phrases.append({
                    "phrase": str(s[0]) if s[0] is not None else "",
                    "start":  int(s[1]) if s[1] is not None else 0,
                    "end":    int(s[2]) if s[2] is not None else 0,
                    "interpretation": (str(s[4]) if len(s) >= 5 and s[4] is not None else ""),
                })

        # Latest PJ
        latest = getattr(mi, "pj_list", [])[-1] if getattr(mi, "pj_list", []) else None
        scores = None
        if latest:
            scores = {
                "accuracy": latest.accuracy_score,
                "clarity": latest.clarity_score,
                "naturalness": latest.naturalness_score,
                "meaning": latest.meaning_similarity,
                "modality": latest.modality_similarity,
                "implication": latest.implication_similarity,
                "syntax": latest.syntax_similarity,
                "context": latest.context_similarity,
                # convert Decimal -> float
                "overall": float(latest.overall_similarity) if latest.overall_similarity is not None else None,
                "quality": float(latest.quality_score) if latest.quality_score is not None else None,
            }

        out.append({
            "sentence": mi.metaphor_sentence or (mi.dataset.sentence if mi.dataset else ""),
            "interpretation": mi.interpretation or "",
            "metaphor_types": mi.metaphor_types or [],
            "metaphor_phrases": slim_phrases,
            "notes": mi.notes or "",
            "scores": scores,
        })

    # Dùng DjangoJSONEncoder để an toàn với Decimal/datetime còn sót
    content = json.dumps(out, ensure_ascii=False, indent=2, cls=DjangoJSONEncoder)
    resp = HttpResponse(content, content_type="application/json; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="ViMUNCH_{timezone.now().date().isoformat()}.json"'
    return resp

# ------------------------- TASK 1 VIEW  ------------------------------

def _norm_types_for_view(v):
    if isinstance(v, str):
        v = [v]
    out = []
    for it in (v or []):
        s = str(it).strip().lower()
        out.append(TYPE_MAP_VI_EN.get(s, s))
    # unique & drop empty
    seen, uniq = set(), []
    for t in out:
        if t and t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


@login_required
def metaphor_interpretation_and_classification_task_view(request):
    user = request.user
    is_admin_user = is_admin(user)

    # 1) Dataset được giao
    if is_admin_user:
        assigned_datasets = MetaphorDataset.objects.all().order_by("id")
    else:
        assigned_ids = TaskAssignment.objects.filter(
            annotator=user,
            task_type="Metaphor Interpretation & Classification",
        ).values_list("datasets__id", flat=True)
        assigned_datasets = MetaphorDataset.objects.filter(id__in=assigned_ids).order_by("id")

    # 2) Chọn dataset hiện tại
    dataset_id = request.GET.get("id")
    if dataset_id:
        dataset = assigned_datasets.filter(id=dataset_id).first()
    else:
        done_ids = (
            MetaphorInterpretation.objects
            .filter(annotator=user)
            .values_list("dataset_id", flat=True)
        )
        dataset = assigned_datasets.exclude(id__in=done_ids).first()

    if not dataset:
        update_task_completion(user, "Metaphor Interpretation & Classification")
        return render(request, "task/no_more_sentences.html")

    # 3) Annotation
    if is_admin_user:
        existing = (
            MetaphorInterpretation.objects
            .filter(dataset=dataset, annotator__isnull=False, version=1)
            .order_by("-id")
            .select_related("annotator")
            .first()
        )
    else:
        existing = MetaphorInterpretation.objects.filter(
            annotator=user, dataset=dataset, version=1
        ).first()

    initial_types, proposed_spans = [], []
    if request.method == "POST":
        form = MetaphorInterpretationForm(request.POST, instance=existing)
        if form.is_valid():
            mi = form.save(commit=False)
            mi.annotator = user
            mi.dataset = dataset
            mi.metaphor_sentence = dataset.sentence
            if not getattr(mi, "version", None):
                mi.version = 1
            mi.task_type = "Metaphor Interpretation & Classification"
            mi.metaphor_phrases = form.cleaned_data.get("metaphor_phrases", []) or []
            mi.metaphor_types = form.cleaned_data.get("metaphor_types", []) or []
            mi.save()

            done_ids = (
                MetaphorInterpretation.objects
                .filter(annotator=user)
                .values_list("dataset_id", flat=True)
            )
            next_ds = assigned_datasets.exclude(id__in=done_ids).exclude(id=dataset.id).first()
            return redirect(f"{request.path}?id={(next_ds.id if next_ds else dataset.id)}")
    else:
        if existing:
            norm = _norm_types_for_view(existing.metaphor_types or [])
            form = MetaphorInterpretationForm(instance=existing, initial={"metaphor_types": norm})
            proposed_spans = existing.metaphor_phrases or []
            initial_types = norm
        else:
            form = MetaphorInterpretationForm()
            proposed_spans = []
            initial_types = []

            # --- NEW: fallback stub ---
            stub = MetaphorInterpretation.objects.filter(
                dataset=dataset, annotator=None, version=1
            ).first()
            if stub:
                proposed_spans = stub.metaphor_phrases or []
                initial_types = _norm_types_for_view(stub.metaphor_types or [])
                form = MetaphorInterpretationForm(initial={"metaphor_types": initial_types})

    # submitted_ids: phân biệt admin vs annotator
    if is_admin_user:
        submitted_ids = (
            MetaphorInterpretation.objects
            .filter(annotator__isnull=False)
            .values_list("dataset_id", flat=True)
        )
    else:
        submitted_ids = (
            MetaphorInterpretation.objects
            .filter(annotator=user)
            .values_list("dataset_id", flat=True)
        )

    return render(
        request,
        "task/metaphor_interpretation_and_classification_task.html",
        {
            "dataset": dataset,
            "assigned_datasets": assigned_datasets,
            "form": form,
            "submitted_ids": list(map(str, submitted_ids)),
            "proposed_spans": proposed_spans,
            "initial_types": initial_types,
            "is_admin_user": is_admin_user,
            "annotator_name": (
                existing.annotator.full_name
                if (is_admin_user and existing and existing.annotator) else None
            ),
        },
    )


# ===================== TASK 2 (Cross-check) =====================

def _pj_assigned_dataset_ids(user) -> set[int]:
    return set(
        TaskAssignment.objects
        .filter(annotator=user, task_type='Paraphrase Judgement')
        .values_list('datasets__id', flat=True)
    )


@login_required
def paraphrase_judgement_input_view(request, task_id=None):
    user = request.user
    is_admin_user = is_admin(user)

    # Pool MI đã hoàn thành (loại stub)
    mi_done_qs = (
        MetaphorInterpretation.objects
        .filter(annotator__isnull=False)
        .filter(MI_DONE_Q())
    )

    if is_admin_user:
        # Admin: xem tất cả MI đã hoàn thành (mỗi dataset lấy MI mới nhất)
        latest_mi_ids = (
            mi_done_qs
            .values('dataset_id')
            .annotate(last_id=Max('id'))
            .values_list('last_id', flat=True)
        )
    else:
        # Annotator: chỉ các dataset đã được GIAO ở PJ, và không tự-chấm bài mình
        assigned_ids = _pj_assigned_dataset_ids(user)
        if not assigned_ids:
            update_task_completion(user, "Paraphrase Judgement")
            return render(request, "task/no_more_sentences.html")

        latest_mi_ids = (
            mi_done_qs
            .exclude(annotator=user)
            .filter(dataset_id__in=assigned_ids)
            .values('dataset_id')
            .annotate(last_id=Max('id'))
            .values_list('last_id', flat=True)
        )

    base_qs = (
        MetaphorInterpretation.objects
        .filter(id__in=latest_mi_ids)
        .select_related("dataset", "annotator")
        .order_by("dataset_id", "id")
    )

    # Đánh dấu đã chấm bởi user hiện tại
    judged_exists = ParaphraseJudgement.objects.filter(
        interpretation=OuterRef("pk"),
        annotator=user
    )
    task_list_qs = base_qs.annotate(is_judged=Exists(judged_exists))

    if not task_list_qs.exists():
        update_task_completion(user, "Paraphrase Judgement")
        return render(request, "task/no_more_sentences.html")

    # Item hiện tại
    if task_id:
        task = get_object_or_404(task_list_qs, pk=task_id)
    else:
        task = task_list_qs.filter(is_judged=False).first() or task_list_qs.first()
        return redirect("task:paraphrase_judgement", task_id=task.id)

    # Bản chấm hiện có của tôi
    existing = ParaphraseJudgement.objects.filter(
        interpretation=task, annotator=user
    ).first()

    # Lưu form
    if request.method == "POST":
        form = ParaphraseJudgementForm(request.POST, instance=existing)
        if form.is_valid():
            j = form.save(commit=False)
            j.annotator = user
            j.interpretation = task
            j.paraphrase_sentence = task.interpretation
            j.save()

            next_task = (
                task_list_qs.filter(is_judged=False)
                .exclude(pk=task.pk)
                .first()
            )
            return redirect("task:paraphrase_judgement", task_id=(next_task.id if next_task else task.id))
    else:
        form = ParaphraseJudgementForm(instance=existing)

    return render(
        request,
        "task/paraphrase_judgement_task.html",
        {
            "task": task,
            "task_list": task_list_qs,
            "form": form,
            "metaphor_text": task.metaphor_sentence,
            "metaphor_spans": task.metaphor_phrases or [],
            "is_admin_user": is_admin_user,
            "annotator_name": (
                task.annotator.full_name
                if (is_admin_user and task.annotator) else None
            ),
        },
    )
