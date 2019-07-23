import numpy as np
import pandas as pd
from predictor_func import similarity_score
from custum_stopwords import custum_stopwords_creat


data = pd.read_csv("test.tsv", sep = "\t")

# data_cust_stop_wor = pd.read_csv("name_of_file.tsv", sep = "\t")
# cus_st = custum_stopwords_creat(data_cust_stop_wor, 10)
for i in range(len(data)):
    print(similarity_score(data['question1'][i], data['question2'][i], [], False))

