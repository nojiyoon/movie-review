from django.urls import path
from . import views
import moviereview_app

urlpatterns = [
    path("", views.IndexView.as_view(), name = "index"),
    path(
        "reviews/<int:review_id>/", 
        views.ReviewDetailView.as_view(), 
        name = "review-detail",
        ),
    path("reviews/new/", views.ReviewCreateView.as_view(), name="review-create"),
    path('review-analyze', moviereview_app.views.index, name='review-analyze')
]