import pandas as pd

def custum_stopwords_creat(data, additional_stopwords = 5):
    
    #Adding stop words to the list
    ques = data['Question_text']
    ques = ques.drop_duplicates()

    import nltk
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    stopwords = nltk.corpus.stopwords.words('english')

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

    for word in word_lis['word'][:additional_stopwords]:
        stopwords.append(word)
    return stopwords
