# ViMUNCH/urls.py
from django.contrib import admin
from django.urls import path, include
from . import views  # Đảm bảo bạn đã import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('task/', include('task.urls')),
    path('users/', include('users.urls')),
    path('', views.home, name='home'),  # Đặt một URL pattern cho trang chủ
]
