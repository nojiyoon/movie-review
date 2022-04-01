import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from wordcloud import WordCloud
from konlpy.tag import Twitter
from collections import Counter
import matplotlib.pyplot as plt


df = pd.read_csv('./data/naver_movie_content.csv')

stopwords = []

tfidf = TfidfVectorizer(stop_words=stopwords)

# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

my_review = {'movie': '이상한 나라의 수학자',
             'sentence': '정말 재미없다. 시간이 아깝다',
             'score': 3}

df.loc[len(df)] = my_review


def get_similar_review(idx=len(df)-1):
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

    # 가장 유사한 10개의 영화의 제목을 리턴한다.
    return df.iloc[idx], df.iloc[movie_indices], review_scores


original_review, similar_review, review_scores = get_similar_review()

similar_review['similarity'] = review_scores

print(similar_review)

# 비슷한 리뷰 10개
text = [similar_review['sentence'].iloc[idx] for idx in range(len(similar_review))]
text = ' '.join(text)

# twitter함수를 통해 읽어들인 내용의 형태소를 분석한다.
okt = Okt()

sentences_tag = []
sentences_tag = okt.pos(text) 
print(sentences_tag)

noun_adj_list = []

# tag가 명사이거나 형용사인 단어들만 noun_adj_list에 넣어준다.
for word, tag in sentences_tag:
    # if tag in ['NNG' , 'VA']:
    if tag in ['Noun' , 'Adjective']: 
        noun_adj_list.append(word)

# 가장 많이 나온 단어부터 40개를 저장한다.
counts = Counter(noun_adj_list)
tags = counts.most_common(40) 
print(tags)

# WordCloud를 생성한다.
# 한글을 분석하기위해 font를 한글로 지정해주어야 된다. macOS는 .otf , window는 .ttf 파일의 위치를
# 지정해준다. (ex. '/Font/GodoM.otf')
wc = WordCloud(font_path='malgun.ttf',background_color="white", max_font_size=60)
cloud = wc.generate_from_frequencies(dict(tags))


# 생성된 WordCloud를 test.jpg로 보낸다.
cloud.to_file('./data/test.jpg')


plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(cloud)
plt.show()