# users/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import User

class UserForm(UserChangeForm):  # dùng cho edit
    class Meta:
        model = User
        fields = ['username', 'full_name', 'email', 'phone', 'sex', 'role', 'expertise', 'experience_years', 'is_active']

class UserCreateForm(UserCreationForm):  # dùng khi add
    class Meta:
        model = User
        fields = ['username', 'password1', 'password2', 'full_name', 'email', 'phone', 'sex', 'role']
