from django.shortcuts import render
from django.views.generic import TemplateView
from pred.serializers import ReviewSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from .forms import ImageForm
from .main import get_similar_review_result, get_review_predict_result

from .models import Review
from .forms import ReviewForm

import json
from django.http import HttpResponse

class IndexView(TemplateView):
    # GET request (index.html를 초기표시) 
    def get(self, req):
        return render(req, 'pred/index.html')

class PredView(APIView):
    def __init__(self):
        self.params = {'result_original': "",
                       'result_similar': "",
                       'result_img': "",
                       'result_sentiment': ""}
    
    def post(self, request):
        my_review = {'movie': request.data['title'],
                    'sentence': request.data['review'],
                    'score': request.data['score']}

        # 결과 반환 
        self.params['result_original'], self.params['result_similar'], self.params['result_img'] = get_similar_review_result(my_review)
        self.params['result_sentiment'] = get_review_predict_result(my_review['sentence'])

        print('original : {}'.format(self.params['result_original']))
        print('긍정이냥 부정이냥 : {}'.format(self.params['result_sentiment']))
        print('이미지 : {}'.format(self.params['result_img']))
        print('비슷한 리뷰들 : {}'.format(self.params['result_similar']))

        # TODO 리턴값은 입맛에 맞게 수정하세용 
        return HttpResponse(json.dumps(self.params, ensure_ascii=False))


# 테스트    
# my_review = {'movie': '스파이더맨',
#             'sentence': '재미있다. 감동적.',
#             'score': 9}

# # 결과 반환 
# original, similar, img = get_similar_review_result(my_review)
# sentiment = get_review_predict_result(my_review['sentence'])


# class PredView(TemplateView):
#     # 생성자
#     def __init__(self):
#         self.params = {'result_original': "",
#                        'result_similar': "",
#                        'result_img': "",
#                        'result_sentiment': "",
#                        'review': ReviewForm()}

#     # GET request (index.html를 초기표시)
#     def get(self, req):
#         return render(req, 'pred/index.html', self.params)

#     # POST request (index.html에 결과를 표시)
#     def post(self, req):
#         # POST 된 Form 데이터 얻기
#         form = ReviewForm(req.POST, req.FILES)
#         # Form data 에러 체크
#         if not input.is_valid():
#             raise ValueError('invalid form')
#         # Form data로 부터 리뷰 얻기
#         title = form.cleaned_data['title']
#         review = form.cleaned_data['review']
#         score = form.cleaned_data['score']
#         # 리뷰를 딕셔너리 형태로 저장
#         my_review = {'title': title,
#                     'review': review,
#                     'score': score}
#         # 결과 반환
#         self.params['result_original'], self.params['result_similar'], self.params['result_img'] = get_similar_review_result(my_review)
#         self.params['result_sentiment'] = get_review_predict_result(my_review['review'])
#         # 페이지에 화면 표시
#         return render(req, 'pred/index.html', self.params)

