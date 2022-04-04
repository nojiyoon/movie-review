from django.urls import path
from .views import PredView
from .views import IndexView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('api/test/', PredView.as_view()),  #TODO API로 사용 안할거면 View 합쳐도 될듯 ㅎ
]