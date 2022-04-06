from django.shortcuts import render
from django.urls import reverse
from django.views.generic import ListView, DetailView, CreateView
# from allauth.account.views import PasswordChangeView
# from django.http import HttpResponse
from moviereview_app.models import Review
from moviereview_app.forms import ReviewForm
from .main import get_review_predict_result, get_similar_review, make_wordcloud
from django.conf import settings
import pandas as pd
from .models import *
from .forms import ReviewForm


data_path = settings.DATA_PATH
df = pd.read_csv(data_path, low_memory=False)

my_review = {'movie': '이상한 나라의 수학자',
             'sentence': '어쩌면 뻔한 스토리일지도 모르지만, 우리가 원하는 세상을 이야기하는 동화. 난롯가처럼 맘이 따뜻해졌습니다.',
             'score': 10}

df.loc[len(df)] = my_review


def index(request):
    if request.method == "POST":
        message = request.POST["message"]
        res = get_review_predict_result(message)
        dataframe = get_similar_review(df, idx=len(df)-1)
        make_wordcloud(dataframe)

        context ={ 'dataframe' : dataframe.to_html(index=False, justify='center'),
                    # 'image_url' :image_path , #wordcloud
                    'text' : res}
                    
        return render(request,'moviereview_app/second.html',context)
    else:
        return render(request,'moviereview_app/first.html') # first.html을 랜더링 


# 제네릭 뷰가 적합한 상황에는 꼭 사용 - 코드가 매우 간결해질 수 있다.
# 모든 로직을 직접 구현하면 코드가 몇 배는 길어졌을 것

class IndexView(ListView):
    model = Review
    template_name = "moviereview_app/index.html"
    context_object_name = "reviews"
    paginate_by = 5
    ordering = ["-dt_created"]

class ReviewDetailView(DetailView):
    model = Review
    template_name = "moviereview_app/review_detail.html"
    pk_url_kwarg = "review_id"

class ReviewCreateView(CreateView):
    model = Review
    form_class = ReviewForm
    template_name = "moviereview_app/review_form.html"

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form) # super() = 상위 클래스

    def get_success_url(self):
        return reverse("review-detail", kwargs={"review_id": self.object.id})


#  class CustomPasswordChangeView(PasswordChangeView):
#      def get_success_url(self):
#          return reverse("index")

# def index(request):
#     return render(request, "moviereview_app/index.html")

# test pandas table
# data = {'name': ['Beomwoo', 'Beomwoo', 'Beomwoo', 'Kim', 'Park'],
#         'year': [2013, 2014, 2015, 2016, 2015],
#         'points': [1.5, 1.7, 3.6, 2.4, 2.9]}
# df = pd.DataFrame(data)