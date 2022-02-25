import pandas as pd
import data.reader as thesisDataReader
from sklearn.feature_extraction.text import TfidfVectorizer

def get_features(corpus, n_gram):
    vectorizer = TfidfVectorizer(ngram_range=(n_gram, n_gram), analyzer='char')
    sparse_matrix = vectorizer.fit_transform(corpus)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=vectorizer.get_feature_names())
    return df


def create_2_gram(corpus):
    return get_features(corpus, 2)

def create_3_gram(corpus):
    return get_features(corpus, 3)

def create_4_gram(corpus):
    return get_features(corpus, 4)

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
