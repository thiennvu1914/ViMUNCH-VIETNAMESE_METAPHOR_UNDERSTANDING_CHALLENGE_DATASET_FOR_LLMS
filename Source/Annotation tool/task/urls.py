from django.urls import path
from . import views

app_name = 'task'

urlpatterns = [
    path('import_data/', views.import_data, name='import_data'),
    path('export-classification/', views.export_classification, name='export_classification_data'),

    # Gộp MI và MC thành một task
    path(
        'metaphor_interpretation_and_classification_task/',
        views.metaphor_interpretation_and_classification_task_view,
        name='metaphor_interpretation_and_classification_task'
    ),

    # Task 2: Paraphrase Judgement
    # Entry không cần task_id: view sẽ tự chọn câu; nếu hết câu -> no_more_sentences.html
    path(
        'paraphrase-judgement/',
        views.paraphrase_judgement_input_view,
        name='paraphrase_judgement_entry'
    ),
    path(
        'paraphrase-judgement/<int:task_id>/',
        views.paraphrase_judgement_input_view,
        name='paraphrase_judgement'
    ),

    path('assign/<str:task_type>/', views.assign_task, name='assign_task'),
    path('task-assignments/', views.task_assignment_list, name='task_assignment_list'),
    path('manage_task/', views.manage_task, name='manage_task'),
    path('annotator/', views.annotator_dashboard, name='annotator_dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
]
