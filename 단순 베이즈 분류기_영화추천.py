import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

review_list =[
    {'review':'This is a great movie. i will watch again', 'type':'positive'},
    {'review':'i like this movie', 'type':'positive'},
    {'review':'amazing movie this year', 'type':'positive'},
    {'review':'cool my boyfriend also said the movie is chool', 'type':'positive'},
    {'review':'awesome of the awesome movie ever', 'type':'positive'},
    {'review':'shame i waste money and time', 'type':'negative'},
    {'review':'regret on this movie. i will never never what movie from this director', 'type':'negative'},
    {'review':'i do not like this movie', 'type':'negative'},
    {'review':'i do not like actors in this movie', 'type':'negative'},
    {'review':'boring boring sleeping movie', 'type':'negative'}]
df = pd.DataFrame(review_list)
df['label'] = df['type'].map({'positive':1, 'negative':0})
df_x = df['review']
df_y = df['label']

cv = CountVectorizer()
x_traincv = cv.fit_transform(df_x)
encoded_input = x_traincv.toarray()
print('\n\벡터 표현n', encoded_input)
print('\n\벡터의 원소 위치별 단어n', cv.get_feature_names())

test_review_list =[
{'review':'This is a great movie. i will watch again', 'type':'positive'},
    {'review':'i like this movie', 'type':'positive'},
    {'review':'amazing movie this year', 'type':'positive'},
    {'review':'cool my boyfriend also said the movie is chool', 'type':'positive'},
    {'review':'awesome of the awesome movie ever', 'type':'positive'},
    {'review':'shame i waste money and time', 'type':'negative'},
    {'review':'regret on this movie. i will never never what movie from this director', 'type':'negative'},
    {'review':'i do not like this movie', 'type':'negative'},
    {'review':'i do not like actors in this movie', 'type':'negative'},
    {'review':'boring boring sleeping movie', 'type':'negative'}]

test_df = pd.DataFrame(test_review_list)
test_df['label'] = test_df['type'].map({'positive':1, 'negative':0})
test_x = test_df['review']
test_y = test_df['label']

x_testcv = cv.transform(test_x)

Mmb = MultinomialNB()
y_train = df_y.astype('int')
Mmb.fit(x_traincv, y_train)
predicted_y = Mmb.predict(x_testcv)
print('\n** ground truth **\n', test_y)
print('\n** 예측치 **\n', predicted_y)
accuracy = accuracy_score(test_y, predicted_y)
print('\n** 정확도 **\n', accuracy)