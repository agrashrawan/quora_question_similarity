#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from nltk.stem.wordnet import WordNetLemmatizer
from keras.models import load_model
from pickle import dump, load
import gensim
import re
import string
from file_to_features import extract_features

model = load_model('my_model_9.h5')
tokenizer = load(open('mymodel_tok', 'rb'))

def similarity_score(q1, q2, costum_stopwords = [], bool_extra_stopwords = False):
    feat = extract_features(q1, q2)

    import nltk
    stopwords = nltk.corpus.stopwords.words('english')
    ##Cleaning data
    if bool_extra_stopwords == True:
        extra_stopwords = gensim.parsing.preprocessing.STOPWORDS
    else:
        extra_stopwords = []

    WNL = WordNetLemmatizer()
    def data_cleaning(text):
        text = text.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")         .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")         .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")         .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")         .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")         .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")         .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
        text = re.sub(r"([0-9]+)000000", r"\1m", text)
        text = re.sub(r"([0-9]+)000", r"\1k", text)
        text = re.sub('[^\w|^\s]', '', text)
        words = []
        text = text.split()
        words = []
        for word in text:
            if word in stopwords:
                continue
            if word in extra_stopwords:
                continue
            if word in costum_stopwords:
                continue
                
            words.append(word)
        for word in text:
            word = WNL.lemmatize(WNL.lemmatize(word, 'n'),'v')
            words.append(word)
        text = " ".join(words)
        return text

    q1 = data_cleaning(q1)
    q2 = data_cleaning(q2)

    is_dupli = 0
    if q1 == q2:
        is_dupli = 1

    q1 = np.expand_dims(q1, axis = 0)
    q2 = np.expand_dims(q2, axis = 0)

    q1s = pad_sequences(tokenizer.texts_to_sequences(q1), maxlen = 25)
    q2s = pad_sequences(tokenizer.texts_to_sequences(q2), maxlen = 25)

    feat_test = np.expand_dims(feat, axis = 0)

    label = np.squeeze(model.predict([q1s, q2s, feat_test]))

    return [is_dupli, label*5.0]
