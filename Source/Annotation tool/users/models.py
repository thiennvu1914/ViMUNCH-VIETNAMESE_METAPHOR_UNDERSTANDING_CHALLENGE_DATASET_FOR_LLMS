# users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    SEX_CHOICES = (
        ('male', 'Male'),
        ('female', 'Female'),
    )
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('senior_annotator', 'Senior Annotator'),
        ('annotator', 'Annotator'),
    )

    full_name = models.CharField(max_length=255)
    sex = models.CharField(max_length=6, choices=SEX_CHOICES)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    expertise = models.TextField(blank=True, null=True)
    experience_years = models.IntegerField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    last_login = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.username
