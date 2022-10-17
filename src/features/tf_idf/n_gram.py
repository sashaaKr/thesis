import pandas as pd
import data.reader as thesisDataReader
from sklearn.feature_extraction.text import TfidfVectorizer

def get_n_gram_range(n_gram):
  n_gram_range = (n_gram, n_gram) if type(n_gram) is int else (n_gram[0], n_gram[1])
  return n_gram_range

def get_features(corpus, n_gram):
  vectorizer = TfidfVectorizer(ngram_range=get_n_gram_range(n_gram), analyzer='char')
  sparse_matrix = vectorizer.fit_transform(corpus)
  doc_term_matrix = sparse_matrix.todense()
  df = pd.DataFrame(doc_term_matrix, columns=vectorizer.get_feature_names())
  return df

class TfIdfFeatures:
  def __init__(self, ngram_range, analyzer):
    self.analyzer = analyzer
    self.ngram_range = ngram_range
    self.vectorizer = TfidfVectorizer(ngram_range = self.ngram_range, analyzer = self.analyzer)
    self.name = str(self.vectorizer)
  
  def get_features(self, corpus):
    sparse_matrix = self.vectorizer.fit_transform(corpus)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns = self.vectorizer.get_feature_names())
    return df

class TfIdf2GramCharFeatures(TfIdfFeatures):
  def __init__(self):
    super().__init__((2,2), 'char')

class TfIdf3GramCharFeatures(TfIdfFeatures):
  def __init__(self):
    super().__init__((3,3), 'char')

class TfIdf4GramCharFeatures(TfIdfFeatures):
  def __init__(self):
    super().__init__((4,4), 'char')
class TfIdf5GramCharFeatures(TfIdfFeatures):
  def __init__(self):
    super().__init__((5,5), 'char')


def create_2_gram(corpus):
  return get_features(corpus, 2)

def create_3_gram(corpus):
  return get_features(corpus, 3)

def create_4_gram(corpus):
  return get_features(corpus, 4)

def create_5_gram(corpus):
  return get_features(corpus, 5)

def create_6_gram(corpus):
  return get_features(corpus, 6)

def create_7_gram(corpus):
  return get_features(corpus, 7)

def create_8_gram(corpus):
  return get_features(corpus, 8)

def create_2_5_gram(corpus):
  return get_features(corpus, (2, 5))

def create_3_5_gram(corpus):
  return get_features(corpus, (3, 5))

def create_zwickau_5_gram():
  corpus = thesisDataReader.get_zwickau_corpus()
  return create_5_gram(corpus)

def create_london_5_gram():
  corpus = thesisDataReader.get_london_corpus()
  return create_5_gram(corpus)

def create_breslau_5_gram():
  corpus = thesisDataReader.get_breslau_corpus()
  return create_5_gram(corpus)
