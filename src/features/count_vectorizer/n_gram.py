import pandas as pd
import data.reader as thesisDataReader
from sklearn.feature_extraction.text import CountVectorizer

def get_features(corpus, n_gram):
    vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), analyzer='char')
    sparse_matrix = vectorizer.fit_transform(corpus)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=vectorizer.get_feature_names())
    return df

# TODO: remove this
def get_ngrams_words_dictionary(text, ngram_from=2, ngram_to=2):    
    vec = CountVectorizer(
        ngram_range = (ngram_from, ngram_to),
        token_pattern = r"(?u)\b\w+\b"
    ).fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis = 0) 
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

    words_freq_dic = {}
    for i in words_freq:
        [word, count] = i
        words_freq_dic[word] = count
    return words_freq_dic

def create_n_gram_corpus(corpus, n_gram):
    vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), analyzer='char')
    vectorizer.fit(corpus)
    X = vectorizer.transform(corpus)
    return vectorizer.inverse_transform(X)


def create_5_gram(corpus):
    return get_features(corpus, 5)

def create_zwickau_5_gram():
    corpus = thesisDataReader.get_zwickau_corpus()
    return create_5_gram(corpus)

def create_london_5_gram():
    corpus = thesisDataReader.get_london_corpus()
    return create_5_gram(corpus)

def create_breslau_5_gram():
    corpus = thesisDataReader.get_brealsu_corpus()
    return create_5_gram(corpus)

def create_zwikau_5_gram_corpus():
    corpus = thesisDataReader.get_zwickau_corpus()
    return create_n_gram_corpus(corpus, 5)

def create_london_5_gram_corpus():
    corpus = thesisDataReader.get_london_corpus()
    return create_n_gram_corpus(corpus, 5)

def create_breslau_5_gram_corpus():
    corpus = thesisDataReader.get_breslau_corpus()
    return create_n_gram_corpus(corpus, 5)