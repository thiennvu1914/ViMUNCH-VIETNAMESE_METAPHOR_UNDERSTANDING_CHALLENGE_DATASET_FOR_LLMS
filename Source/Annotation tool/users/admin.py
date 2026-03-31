from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User
from .forms import UserCreateForm, UserForm

class UserAdmin(BaseUserAdmin):
    add_form = UserCreateForm
    form = UserForm
    model = User

    list_display = ('username', 'full_name', 'email', 'phone', 'role', 'is_active')
    list_filter = ('role', 'is_active')
    search_fields = ('username', 'full_name', 'email')
    ordering = ('username',)

    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Thông tin cá nhân', {'fields': ('full_name', 'email', 'phone', 'sex', 'role', 'expertise', 'experience_years')}),
        ('Quyền truy cập', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'password1', 'password2', 'full_name', 'email', 'phone', 'sex', 'role', 'is_active')}
        ),
    )

admin.site.register(User, UserAdmin)
