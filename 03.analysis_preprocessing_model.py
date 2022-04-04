import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request

from hanspell import spell_checker
import konlpy
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import io
from konlpy.utils import pprint
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import urllib.request 
import time

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPool1D, MaxPooling2D, Flatten, Conv2D, MaxPooling1D, Conv1D
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout # Permute : 주어진 패턴에 따라서 입력 차수 변경
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


review = pd.read_csv('naver_movie_content.csv', low_memory=False, encoding = 'UTF-8')

# 긍부정 라벨링하기
# 긍정이 1, 부정이 0
# 8점 이상을 긍정, 8점 미만을 부정
review['label'] = 0
review.loc[review['score'] >= 8, 'label'] = 1

# py-hanspell(맞춤법, 띄어쓰기 검사기) 적용하기
review = review.dropna(how = 'any') 

new_sent = []
def review_sent(text):
    for i in range(len(text)):
        spelled_sent = spell_checker.check(text[i])
        hanspell_sent = spelled_sent.checked
        sent = hanspell_sent
        new_sent.append(sent)

review['sentence'] = new_sent
df_review = review['sentence']

# 전처리
#영어 소문자로 모두 변환
df_review = df_review.str.lower()

#반복되는 문자 제거하기
df_review = df_review.str.replace("ㅋ","") # ㅋㅋㅋ제거
df_review = df_review.str.replace("ㅜ","") # ㅜㅜㅜ제거
df_review = df_review.str.replace("ㅠ","") # ㅠㅠㅠ제거
df_review = df_review.str.replace("ㅎ","") # ㅎㅎㅎ제거
df_review = df_review.str.replace("ㅇ","") # ㅇㅇ제거
df_review = df_review.str.replace("ㄷ","") # ㄷㄷ제거
df_review = df_review.str.replace("ㄱ","") # ㄱㄱ제거
df_review = df_review.str.replace("-","") # --제거
df_review = df_review.str.replace("_","") # __제거
df_review = df_review.str.replace("ㅡ","") # ㅡㅡ제거
df_review = df_review.str.replace("ㅉ","")
df_review = df_review.str.replace("ㅆㅈ","")
df_review = df_review.str.replace("ㅏ","")
df_review = df_review.str.replace("ㅡ","")

#인터넷 줄임말 단어 변환해주기
df_review = df_review.str.replace("ㄱㅊ","괜찮")
df_review = df_review.str.replace("ㅈㄹ","지랄")
df_review = df_review.str.replace("ㅈㅁ","정말")
df_review = df_review.str.replace("ㅇㅈ","인정")
df_review = df_review.str.replace("ㄹㅇ","정말")
df_review = df_review.str.replace("ㅈㅅ","죄송")
df_review = df_review.str.replace("좋ㅈ었습니다","좋았습니다")
df_review = df_review.str.replace("ㅈㄴ","존나")

#영어표현 변경
df_review = df_review.str.replace("cg","씨지")
df_review = df_review.str.replace("ost","오에스티")
df_review = df_review.str.replace("sf","공상과학")
df_review = df_review.str.replace("tv","텔레비전")
df_review = df_review.str.replace("sns","소셜미디어")
df_review = df_review.str.replace("good","좋다")
df_review = df_review.str.replace("best","좋다")

#불용어 삭제
df_review = df_review.str.replace("영화","")

#정규표현식 
df_review = df_review.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 한글과 공백을 제외하고 모두 제거
df_review = df_review.str.replace('^ +', "") #공백은 empty값으로 변경
#df_review.drop_duplicates(subset = ['content'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
df_review.replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
df_review = df_review.dropna(how='any') # Null 값 제거


#전처리 데이터 다시 넣어주기
review['sentence'] = df_review

# Null 값이 한개라도 존재하는 행 제거
review = review.dropna(how = 'any') 


# 형태소 분석기_Okt 사용
okt = Okt()

#리뷰 데이터 형태소분석 및 태깅
review_tag = []

for sentence in tqdm(review['sentence']):
    tokenized_sentence = okt.pos(sentence, stem = True) # 토큰화
    review_tag.append(tokenized_sentence)

review['tagging'] = review_tag

tag = review["tagging"]

review_new = []

for sentence in tqdm(review['sentence']):
    tokenized_sentence = okt.pos(sentence, stem = True) # 토큰화
    words_list_sentence = [word for word, tag in tokenized_sentence if tag in ['Noun','Adjective','Verb']] 
    review_new.append(words_list_sentence)

review['tag_list'] = review_new

#단어 카운트하기
from collections import Counter
counts = Counter(words_list)


# Modeling
# Train data, Test data split
train_data = review['tag_list']
test_data = review['label'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.2, random_state = 422)


# 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 전체 단어 개수 중 빈도수 1 이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# 빈 샘플 제거
# 샘플들의 길이를 확인해서 길이가 0인 샘플들의 인덱스 추출
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
len(drop_train)

# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
Y_train = np.delete(Y_train, drop_train, axis=0)

def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1

# 최대 길이가 133이므로 만약 30으로 패딩할 경우, 몇 개의 샘플들을 온전히 보전할 수 있는지 확인
max_len = 30
# below_threshold_len(max_len, X_train)


# Padding
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)


# Create Model
# CNN를 사용한 모델 (model1) 정의
embedding_dim = 100
hidden_units = 128

model1 = Sequential()
model1.add(Embedding(vocab_size, embedding_dim, input_length = max_len))
model1.add(Conv1D(256, kernel_size = 3, padding = 'SAME', activation = 'relu', input_shape = (10, 4)))
model1.add(Conv1D(128, kernel_size = 3, padding = 'SAME', activation = 'relu'))
model1.add(Conv1D(64, kernel_size = 3, padding = 'SAME', activation = 'relu'))
model1.add(MaxPooling1D(pool_size = 2))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dropout(0.3))
model1.add(Dense(100, activation = 'relu'))
model1.add(Dense(32, activation = 'relu'))
model1.add(Dense(1, activation = 'sigmoid'))
    
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
mc = ModelCheckpoint('./model/review_best_model_10.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model1.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc']) # adam
model1.summary()


# LSTM를 사용한 모델 (model2) 정의
embedding_dim = 100
hidden_units = 128

model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_dim))
model2.add(LSTM(hidden_units))
model2.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
mc = ModelCheckpoint('./model/review_best_model_8.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model2.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc']) # adam
model2.summary()


# fit
r1 = model1.fit(X_train, Y_train, epochs = 30, callbacks = [es, mc], batch_size = 64, validation_data = (X_test, Y_test))

r2 = model2.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 64, validation_data = (X_test, Y_test))


loaded_model = load_model('./model/review_best_model_10.h5') # 정확도 85.51%
loaded_model.evaluate(X_test, Y_test)


loaded_model = load_model('./model/review_best_model_8.h5') # 정확도 85.29%
loaded_model.evaluate(X_test, Y_test)


# Evaluate the Model

import matplotlib.pyplot as plt
print(r1.history.keys())

plt.plot(r1.history['acc'])
plt.plot(r1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


# 리뷰 예측

review1 = "솔직히 두 주인공의 감정이입이 안되다보니 아무느낌 없습니다"


def review_predict(review1):
    
    # 1. spell_checker
    spelled_sent = spell_checker.check(review1)
    hanspell_sent = spelled_sent.checked
    new_review = hanspell_sent
    
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
    padding_review = pad_sequences(new_review, maxlen = max_len)

    # 9. predcit 
    score = float(loaded_model.predict(padding_review))

    # 10. 결과 값 판단 
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰".format((1- score) * 100))


review_predict("솔직히 두 주인공의 감정이입이 안되다보니 아무느낌 없습니다")
# sequential -> 96.00% 확률로 부정 리뷰

# review_predict("정겨운 사람과 따뜻한 공간을 떠나온 지금도 가슴 속에 추억한다.")
# # sequential -> 99.76% 확률로 긍정 리뷰

# review_predict("멸공이라는 단어를 악으로 만든 영화.전형적인 한국식 정치영화")
# # sequential -> 63% 확률로 부정 리뷰