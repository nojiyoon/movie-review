from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from django.views.generic import TemplateView
# from moviereview_app.views import CustomPasswordChangeView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('moviereview_app.urls')),
    path('', include('allauth.urls')),
]

urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)