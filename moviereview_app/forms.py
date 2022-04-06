from socket import fromshare
from django import forms
from .models import Review, User

class SignupForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["email"]

    def signup(self, request, user):
        user.email = self.cleaned_data["email"]
        user.save()

class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = [
            "title",
            "movie_name",
            "movie_link",
            "rating",
            "image1",
            "image2",
            "image3",
            "content",
        ]

        widgets = {
            "rating" : forms.RadioSelect,
        }
        # 드롭다운 대신 라디오 버튼 선택