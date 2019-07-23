#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from utils import *
import random
import pandas as pd
import tensorflow as tf
import spacy
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout,Bidirectional ,LSTM,Add, Activation, Subtract, Dot, Multiply, Lambda, Concatenate, BatchNormalization
from keras.layers import add, multiply, concatenate, GaussianNoise
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.activations import exponential
from keras.backend import constant, sum
from keras.utils import to_categorical
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
np.random.seed(1)
import re
import string
from file_to_features import extract_features


input_file_name = "train.tsv"
extract_features(input_file_name)
print("Done")

input_file_name = "test.tsv"
extract_features(input_file_name)
print("Done")


# In[2]:


##Cleaning data

WNL = WordNetLemmatizer()
stopwor = set(stopwords.words('english'))
def data_cleaning(text):
    text = text.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")         .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")         .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")         .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")         .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")         .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")         .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    text = re.sub('[^\w|^\s]', '', text)
    words = []
    text = text.split()
    for word in text:
        word = WNL.lemmatize(WNL.lemmatize(word, 'n'),'v')
        words.append(word)
    text = " ".join(words)
    return text
round1 = lambda x: data_cleaning(x)


# In[3]:


### Reads and clean data
data = pd.read_csv("train.tsv", sep = '\t')
data = data.fillna('')

clean_data_old = pd.DataFrame([data['id'],data['qid1'],data['qid2'],data['question1'].apply(round1), data['question2'].apply(round1),data['is_duplicate']])# columns= ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'] )
clean_data_old = clean_data_old.transpose()


# In[4]:


#Adding stop words to the list
ques = clean_data_old['question1']
ques = ques.append(clean_data_old['question2'])
ques = ques.drop_duplicates()
print(len(ques))

import nltk
stopwords_set = set(nltk.corpus.stopwords.words('english'))
stopwords = nltk.corpus.stopwords.words('english')
print stopwords

words = {}
for que in ques:
    text = que.split()
    for word in text:
        if word in stopwords_set:
            continue;
        if words.get(word) is None:
            words[word] = 1
        else:
            words[word] = words[word] + 1
            
word_lis = []
for word, no in words.items():
    word_lis.append([word, no])
word_lis = pd.DataFrame(word_lis, columns = ["word", "freq"])
word_lis = word_lis.sort_values(by = "freq", ascending=False)

for word in word_lis['word'][:20]:
    stopwords.append(word)
# print stopwords


# In[5]:


## Remove stop words

def s_wor_rm(text):
    words = []
    text = text.split()
    for word in text:
        if word in stopwords:
            continue
            
        words.append(word)
    text = " ".join(words)
    return text
round2 = lambda text: s_wor_rm(text)


# In[6]:


clean_data = pd.DataFrame([clean_data_old['id'],clean_data_old['qid1'],clean_data_old['qid2'],
                           clean_data_old['question1'].apply(round2), clean_data_old['question2'].apply(round2),
                           clean_data_old['is_duplicate']])
# columns= ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'] )
clean_data = clean_data.transpose()


# In[7]:


## Makes word to vec dict

glove_vec_file = open("glove.6B.300d.txt","r")

words = []
word_to_vec = {}
for line in glove_vec_file:
    line = line.strip().split()
    vec = np.array([val for val in line[1:]], dtype = 'float32')
    words.append(line[0])
    word_to_vec[line[0]] = vec


# In[8]:


## Tokenizes words....................

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.array(words))
word_index = tokenizer.word_index


# In[9]:
#Used to save tokenizer.

from pickle import dump, load
dump(tokenizer, open('mymodel_tok', 'wb'))


# In[9]:
# It will convert word_to_vec file in keras embedding layer.

def pretrained_model(word_index, word_to_vec):
    dim = len(word_to_vec['the'])
    vocab_len = len(word_index) + 1
    emb_mat = np.zeros((vocab_len, dim))
    for word, i in word_index.items():
        if word_to_vec.get(word) is not None:
            emb_mat[i,:] = word_to_vec[word]
    print(emb_mat.shape)
    embedding_layer = Embedding(vocab_len, dim, trainable = False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_mat])
    return embedding_layer


# In[10]:


def nn_model(max_len,feat, word_index, word_to_vec):
    sequ1 = Input(shape = (max_len,), dtype = "float32")
    sequ2 = Input(shape = (max_len,), dtype = "float32")
    f_train = Input(shape = (feat,), dtype= "float32")
    embedding_lay = pretrained_model(word_index, word_to_vec)

    lstm_layer = LSTM(75, recurrent_dropout=0.2)
    
    seq1 = embedding_lay(sequ1)
    seq2 = embedding_lay(sequ2)

    x1 = lstm_layer(seq1)

    y1 = lstm_layer(seq2)

#     features_input = Input(shape=(f_train.shape[1],), dtype="float32")
    features_dense = BatchNormalization()(f_train)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(1, activation="sigmoid")(merged)

    
    
    model = Model(inputs = [sequ1, sequ2, f_train], outputs = out)
    return model


# In[11]:


model = nn_model(25,15, word_index, word_to_vec)
model.summary()


# In[12]:


model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])


# In[13]:


q1s = pad_sequences(tokenizer.texts_to_sequences(clean_data['question1']), maxlen = 25)
q2s = pad_sequences(tokenizer.texts_to_sequences(clean_data['question2']), maxlen = 25)
label = clean_data['is_duplicate']
feature = pd.read_csv('nlp_features_train.tsv', sep = "\t")
# feature = feature.dropna()
# len(feature[0])


# In[14]:


data_test = pd.read_csv("test.tsv", sep = '\t')
data_test = data_test.fillna('')
# clean_data = pd.DataFrame(data['question1'].apply(round1),data['question2'].apply(round1))
clean_data_test = pd.DataFrame([data_test['id'],data_test['qid1'],data_test['qid2'],data_test['question1'].apply(round1).apply(round2), data_test['question2'].apply(round1).apply(round2),data_test['is_duplicate']])# columns= ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'] )
clean_data_test = clean_data_test.transpose()

q1s_t = pad_sequences(tokenizer.texts_to_sequences(clean_data_test['question1']), maxlen = 25)
q2s_t = pad_sequences(tokenizer.texts_to_sequences(clean_data_test['question2']), maxlen = 25)
label_t = clean_data_test['is_duplicate']
feat_test = pd.read_csv("nlp_features_test.tsv", sep = '\t')


# In[15]:


for i in range(10):
    model.fit([q1s, q2s, feature], label, epochs = 1, batch_size = 400, shuffle=True)
    stri = "my_model_" + str(i) + ".h5"
    model.save(stri)
    loss, acc = model.evaluate([q1s_t, q2s_t, feat_test], label_t)
    print([loss, acc])

