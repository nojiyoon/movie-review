import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


okt = Okt()

loaded_model = load_model('./model/sentiment_test.h5')

with open('./model/tokenizer_mecab_test.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

stopwords = []


def review_predict(review1):
    # 1 한글 데이터가 아닌 값들 제거 
    new_review = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', review1)
    # 2. 토큰화
    new_review = okt.morphs(new_review, stem=True)
#   new_review = okt.morphs(new_review, stem=True)
#   new_review = mecab.morphs(new_review)
#    print(new_review)
    # 3. 불용어 제거
    new_review = [word for word in new_review if not word in stopwords]
#     print(new_review)
    # 4. Embedding 처리 (text -> sequences)
    new_review = tokenizer.texts_to_sequences([new_review])
#    print(new_review)
    # 5. padding -> 길이 값을 맞추기 위해 
    padding_review = pad_sequences(new_review, maxlen=30)
#    print(padding_review)
    # 6. predcit 
    score = float(loaded_model.predict(padding_review))
#     print(score)
    # 7. 결과 값 판단 
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰".format((1- score) * 100))


review_predict('감동적')