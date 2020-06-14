import pyprind
import pandas as pd
from string import punctuation
import numpy as np

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head())

from collections import Counter
counts = Counter()
pbar = pyprind.ProgBar(len(df['review']), title='단어 발생 횟수')

for i, review in enumerate(df['review']):
    text = ",".join([c if c not in punctuation else ''+c+'' for c in review]).lower()
    df.loc[i, 'review'] = text
    pbar.update()
    counts.update(text.split())
print(df.head(1))

word_counts = sorted(counts, key=counts.get, reverse=True)
word_to_int = {word: idx for idx, word in enumerate(word_counts, 1)}
mapped_reviews = []

pbar = pyprind.ProgBar(len(df['review']), title='reviewml의 정수열 반환')

for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()

print(mapped_reviews[0])
sequence_length = 200
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype= int)

for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i,-len(row):] = review_arr[-sequence_length:]

X_train = sequences[:37500,:]
y_train = df.loc[:37500, 'sentiment'].values
X_test = sequences[37500:,:]
y_test = df.loc[37500:, 'sentiment'].values

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
n_words = len(word_to_int)+1
print('no. of distinct words : ', n_words)

from tensorflow.keras import models, layers
model = models.Sequential()
model.add(layers.Embedding(n_words, 200, embeddings_regularizer='l2'))#l or I
print(model.summary())

model.add(layers.LSTM(16))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

callback_list = [ModelCheckpoint(filepath='sentiment_rnn_checkpoint.k5', monitor='val_loss',
                                 save_best_only=True), TensorBoard(log_dir='sentiment_rnn_logs\{}'.format(int(time.time()%100)))]

history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.3, callbacks=callback_list)

import matplotlib.pyplot as plt
epochs = np.arange(1,11)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

model.load_weights('sentiment_rnn_checkpoint.h5')

print(model.predict_classes(X_test[:10]))
print(model.predict(X_test[:10]))
print((model.predict(X_test[:10])>0.5).astype('int32'))