import re
import nltk
from sklearn import preprocessing

import pandas as pd
import numpy as np
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
#from keras import backend as K
embed_size = 100 # how big is each word vector
max_features = 25000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
nb_classes=2
EMBEDDING_FILE='glove.6B.100d.txt'
#TRAIN_DATA_FILE='liwc_input.csv'
#TEST_DATA_FILE='liwc_test.csv'
train = pd.read_csv(r"liwc_input.csv")
test = pd.read_csv(r"liwc_test.csv")
list_sentences_train = train["text"].fillna("_na_").values
y_train = np.array(train['rating'])
y_test = np.array(test['rating'])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

list_sentences_test = test["text"].fillna("_na_").values
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
unigrams = pad_sequences(list_tokenized_train, maxlen=maxlen)
unigrams_t = pad_sequences(list_tokenized_test, maxlen=maxlen)
liwc_scaler = preprocessing.StandardScaler()
liwc=liwc_scaler.fit_transform(train.ix[:, "WC":"OtherP"])
liwc_t = liwc_scaler.transform(test.ix[:, "WC":"OtherP"])
X_t = np.hstack(unigrams)
X_te = np.hstack(unigrams_t)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(nb_classes, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(unigrams, Y_train, batch_size=32, epochs=10) # validation_split=0.1);
y_test = np.argmax(model.predict(unigrams_t, batch_size=1024, verbose=0),axis=1)
print('prediction 7 accuracy: ', accuracy_score(test['rating'], y_test))
