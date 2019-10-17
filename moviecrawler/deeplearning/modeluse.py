import tensorflow as tf
import numpy as np
from konlpy.tag import Okt
from pymongo import MongoClient
from pprint import pprint

# 1. MongoDB Connection
client = MongoClient('localhost', 27017) #ip주소, port 번호
db = client['local'] # 'local' Database를 선택
collection = db.movie # Collection 선택
collection = db.get_collection('movie') # 동적으로 Collection 선택

# 2. MongoDB 데이터 불러오기
reply_list = []
for one in collection.find({}, {'_id' : 0, 'contents': 1}):
    reply_list.append(one['contents'])


#print('>> 댓글 내용')
#pprint(reply_list)
#pprint('count', len(reply_list))

# 3. Okt() 형태소 분석기 객체 생성
okt = Okt()


# 4. selectword.txt 불러오기
def read_data(filename):
    with open(filename,'r',encoding='UTF-8') as f:
        data =[]
        while True:
            line =f.readline()[:-1]
            if not line: break
            data.append(line)
    return data


select_words =read_data('selectword.txt')
# print(select_wores[:10])
# 모델 블라억;
model = tf.keras.models.load_model('my_model.h5')


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def term_frequency(doc):
    return [doc.count(word) for word in select_words]


# 예측하는 함수 구현하기
def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))

    if(score > 0.5):
        print('[{}]는 {:.2f}% 확률로 긍정 리뷰에 가깝습니다:)'.format(review, score*100))
    else:
        print('[{}]는 {:.2f}% 확률로 부정 리뷰에 가깝습니다:('.format(review, (1-score)*100))


for one in reply_list:
    predict_pos_neg(one)

