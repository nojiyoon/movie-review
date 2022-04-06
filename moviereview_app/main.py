# from keras.backend import tensorflow_backend as backend
import tensorflow.keras.backend as backend #-> tensorflow 2.0에서 변경
from django.conf import settings
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from wordcloud import WordCloud
from collections import Counter
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import cv2
from PIL import Image
import numpy as np


model_file_path = settings.MODEL_FILE_PATH
loaded_model = load_model(model_file_path)

pickle_path = settings.PICKLE_FILE_PATH
with open(pickle_path, 'rb') as f:
    tokenizer = pickle.load(f)

data_path = settings.DATA_PATH
df = pd.read_csv(data_path, low_memory=False)

tokenizer = Tokenizer(500)
stopwords = []
okt = Okt()
tfidf = TfidfVectorizer(stop_words=stopwords)


def get_similar_review(df, idx=len(df)-1): 

    tfidf_matrix = tfidf.fit_transform(df['sentence'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 해당 리뷰와 모든 리뷰와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]
    
    # 유사도 점수를 얻는다
    review_scores = [idx[1].round(2) for idx in sim_scores]

    similar_review = df.iloc[movie_indices]

    similar_review['similarity'] = review_scores

    similar_review = similar_review[['movie', 'sentence', 'score', 'similarity']]

    return similar_review


def get_review_predict_result(review1):
    # 1. spell_checker (생략)
    # spelled_sent = spell_checker.check(review1)
    # hanspell_sent = spelled_sent.checked
    # new_review = hanspell_sent
    new_review = review1

    # 2. 반복되는 문자 제거하기
    new_review = new_review.replace("ㅋ","") # ㅋㅋㅋ제거
    new_review = new_review.replace("ㅜ","") # ㅜㅜㅜ제거
    new_review = new_review.replace("ㅠ","") # ㅠㅠㅠ제거
    new_review = new_review.replace("ㅎ","") # ㅎㅎㅎ제거
    new_review = new_review.replace("ㅇ","") # ㅇㅇ제거
    new_review = new_review.replace("ㄷ","") # ㄷㄷ제거
    new_review = new_review.replace("ㄱ","") # ㄱㄱ제거
    new_review = new_review.replace("-","") # --제거
    new_review = new_review.replace("_","") # __제거
    new_review = new_review.replace("ㅡ","") # ㅡㅡ제거

    new_review = new_review.replace("ㅉ","")
    new_review = new_review.replace("ㅆㅈ","")
    new_review = new_review.replace("ㅏ","")
    new_review = new_review.replace("ㅡ","")
    
    # 3. 단어 변환해주기 - 줄임말 변경
    new_review = new_review.replace("ㄱㅊ","괜찮")
    new_review = new_review.replace("ㅈㄹ","지랄")
    new_review = new_review.replace("ㅈㅁ","정말")
    new_review = new_review.replace("ㅇㅈ","인정")
    new_review = new_review.replace("ㄹㅇ","정말")
    new_review = new_review.replace("ㅈㅅ","죄송")
    new_review = new_review.replace("좋ㅈ었습니다","좋았습니다")
    new_review = new_review.replace("ㅈㄴ","존나")

    # 4. 영어표현 변경
    new_review = new_review.replace("cg","씨지")
    new_review = new_review.replace("ost","오에스티")
    new_review = new_review.replace("sf","공상과학")
    new_review = new_review.replace("tv","텔레비전")
    new_review = new_review.replace("sns","소셜미디어")
    new_review = new_review.replace("good","좋다")
    new_review = new_review.replace("best","좋다")
    
    # 5. 한글과 공백을 제외하고 모두 제거 
    new_review = new_review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

    # 6. 토큰화
    new_review = okt.pos(new_review, stem = True) 
    new_review = [word for word, tag in new_review if tag in ['Noun','Adjective','Verb']] 
    
    # 7. Embedding 처리 (text -> sequences)
    new_review = tokenizer.texts_to_sequences([new_review])

    # 8. padding -> 길이 값을 맞추기 위해 
    padding_review = pad_sequences(new_review, maxlen = 30)

    # 9. predcit 
    score = float(loaded_model.predict(padding_review))
    
    # 10. 결과 값 판단 
    if score > 0.5:
        # print("{:.2f}% 확률로 긍정 리뷰".format(score * 100))
        conclusion = "{:.2f}% 확률로 긍정 리뷰".format(score * 100)
    else:
        # print("{:.2f}% 확률로 부정 리뷰".format((1- score) * 100))
        conclusion = "{:.2f}% 확률로 부정 리뷰".format((1- score) * 100)   

    return conclusion


def make_wordcloud(similar_review):

    # 비슷한 리뷰 10개
    text = [similar_review['sentence'].iloc[idx] for idx in range(len(similar_review))]
    text = ' '.join(text)

    # twitter함수를 통해 읽어들인 내용의 형태소를 분석한다.
    okt = Okt()

    sentences_tag = []
    sentences_tag = okt.pos(text) 
    # print(sentences_tag)

    noun_adj_verb_list = []

    # tag가 명사이거나 형용사인 단어들만 noun_adj_verb_list에 넣어준다.
    for word, tag in sentences_tag:
        if tag in ['Noun' , 'Adjective', 'Verb']: 
            noun_adj_verb_list.append(word)

    # 가장 많이 나온 단어부터 40개를 저장한다.
    counts = Counter(noun_adj_verb_list)
    tags = counts.most_common(40) 
    # print(tags)

    # WordCloud를 생성한다.
    mask = Image.new("RGBA",(700,600), (255,255,255)) #(2555,2575)는 사진 크기, (255,255,255)는 색을의미
    image = Image.open(settings.MEDIA_ROOT + '/heart.png').convert("RGBA")
    x,y = image.size
    mask.paste(image,(0,0,x,y),image)
    mask = np.array(mask)

    wordcloud = WordCloud(font_path='malgun.ttf',
                        background_color ='white', colormap='autumn',
                        width = 100, height = 100, random_state = 43, mask = mask,
                        prefer_horizontal = True).generate_from_frequencies(dict(tags))

    # 생성된 WordCloud를 test.jpg로 보낸다.
    wordcloud.to_file( settings.MEDIA_ROOT + '/test3.jpg')
    image = cv2.imread(settings.MEDIA_ROOT + '/test3.jpg')

    return image