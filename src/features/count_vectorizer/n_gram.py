import pandas as pd
import data.reader as thesisDataReader
from sklearn.feature_extraction.text import CountVectorizer

def get_features(corpus, n_gram):
    vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), analyzer='char')
    sparse_matrix = vectorizer.fit_transform(corpus)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=vectorizer.get_feature_names())
    return df

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