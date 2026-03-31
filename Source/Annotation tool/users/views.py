from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required, user_passes_test
from .models import User
from .forms import UserForm
from .forms import UserForm, UserCreateForm



# Kiểm tra xem người dùng có phải là admin hay không
def is_admin(user):
    return user.is_superuser

# Đăng nhập
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        # Kiểm tra xem username có tồn tại không
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Chuyển hướng sau khi đăng nhập thành công
        else:
            # Nếu mật khẩu sai hoặc không tìm thấy người dùng
            messages.error(request, 'Tên đăng nhập hoặc mật khẩu không đúng.')
            return render(request, 'users/login.html')
    
    return render(request, 'users/login.html')  # Render form đăng nhập khi chưa submit

# Đăng xuất
def logout_view(request):
    logout(request)
    return redirect('users:login')

# ================================== PROFILE ==============================================
@login_required
def profile_view(request):
    # Chỉ hiển thị thông tin người dùng hiện tại
    user = request.user
    return render(request, 'users/profile.html', {'user': user})

# Xem chi tiết annotator
@login_required
@user_passes_test(is_admin)
def annotator_detail(request, user_id):
    try:
        annotator = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, "Annotator không tồn tại.")
        return redirect('users:annotator_list')
    return render(request, 'users/annotator_detail.html', {'annotator': annotator})

# Hàm chỉnh sửa thông tin cá nhân
@login_required
def edit_profile(request):
    user = request.user
    if request.method == 'POST':
        # Cập nhật thông tin người dùng
        user.full_name = request.POST.get('full_name', user.full_name)
        user.email = request.POST.get('email', user.email)
        user.phone = request.POST.get('phone', user.phone)
        user.expertise = request.POST.get('expertise', user.expertise)
        user.experience_years = request.POST.get('experience_years', user.experience_years)
        user.save()
        messages.success(request, 'Thông tin cá nhân đã được cập nhật!')
        return redirect('users:profile')  # Sau khi cập nhật, chuyển hướng về trang cá nhân

    return render(request, 'users/edit_profile.html', {'user': user})

# Đổi mật khẩu
@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)  # Đảm bảo không bị thoát phiên
            messages.success(request, 'Mật khẩu đã được thay đổi thành công.')
            return redirect('users:profile')
        else:
            messages.error(request, 'Có lỗi xảy ra khi thay đổi mật khẩu.')
    else:
        form = PasswordChangeForm(user=request.user)
    return render(request, 'users/change_password.html', {'form': form})


# ================================== Annotators Management ===========================================
@login_required
@user_passes_test(is_admin)
def annotator_list(request):
    annotators = User.objects.all()  # Hoặc có thể thêm điều kiện lọc nếu cần
    return render(request, 'users/annotator_list.html', {'annotators': annotators})

# Thêm annotator mới
@login_required
@user_passes_test(is_admin)
def add_annotator(request):
    if request.method == 'POST':
        form = UserCreateForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.role = 'annotator'  # đảm bảo role được set
            user.set_password(form.cleaned_data['password1'])  # mã hóa mật khẩu
            user.save()
            messages.success(request, 'Thêm annotator thành công!')
            return redirect('users:annotator_list')
        else:
            messages.error(request, 'Lỗi khi thêm annotator.')
    else:
        form = UserCreateForm()
    return render(request, 'users/add_annotator.html', {'form': form})

# Chỉnh sửa annotator
@login_required
@user_passes_test(is_admin)
def edit_annotator(request, user_id):
    try:
        annotator = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, "Annotator không tồn tại.")
        return redirect('users:annotator_list')

    if request.method == 'POST':
        form = UserForm(request.POST, instance=annotator)
        if form.is_valid():
            form.save()
            messages.success(request, 'Chỉnh sửa thông tin annotator thành công!')
            return redirect('users:annotator_list')
        else:
            messages.error(request, 'Lỗi khi chỉnh sửa annotator.')
    else:
        form = UserForm(instance=annotator)

    return render(request, 'users/edit_annotator.html', {'form': form, 'annotator': annotator})

# Xóa annotator
@login_required
@user_passes_test(is_admin)
def delete_annotator(request, user_id):
    try:
        annotator = User.objects.get(id=user_id)
        annotator.delete()
        messages.success(request, 'Đã xóa annotator!')
    except User.DoesNotExist:
        messages.error(request, "Annotator không tồn tại.")
    
    return redirect('users:annotator_list')
