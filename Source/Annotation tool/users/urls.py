# users/urls.py
from django.urls import path
from .views import login_view, logout_view, profile_view, edit_profile, change_password
from .views import annotator_list, add_annotator, edit_annotator, delete_annotator, annotator_detail

app_name = 'users'

urlpatterns = [
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    
    path('profile/', profile_view, name='profile'),
    path('profile/edit/', edit_profile, name='edit_profile'),
    path('change-password/', change_password, name='change_password'),
    
    path('annotators/', annotator_list, name='annotator_list'),
    path('annotator/add/', add_annotator, name='add_annotator'),
    path('annotator/edit/<int:user_id>/', edit_annotator, name='edit_annotator'),
    path('annotator/delete/<int:user_id>/', delete_annotator, name='delete_annotator'),
    path('annotators/detail/<int:user_id>/', annotator_detail, name='annotator_detail'),
]