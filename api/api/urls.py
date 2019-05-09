from django.contrib import admin
from django.conf.urls import include
from django.urls import path

urlpatterns = [
    path('classifier/', include('classifier.urls')),
    #path('admin/', admin.site.urls)
]