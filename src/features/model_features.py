import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LONDON_VERSION_LABEL = 0
ZWICKAU_VERSION_LABEL = 1
BURCHARD_VERSION_LABEL = 2

METADATA_COLUMN_NAMES = ['index', 'version']

def create_p_features(vectorizer, p, p_index, corpus_label):
  p_features_vectorizer = vectorizer.transform([p]).toarray()[0]
  p_features_extended = np.append(p_features_vectorizer, [p_index, corpus_label])
  return p_features_extended

def create_corpus_features(vectorizer, corpus, corpus_label):
  data = []

  for index, p in enumerate(corpus):
    p_features = create_p_features(vectorizer, p, index, corpus_label)
    data.append(p_features)
  
  return data

def create_tf_idf_vectorizer(corpus, n_gram = 5, analyzer='char'):
  vectorizer = TfidfVectorizer(ngram_range=(n_gram, n_gram), analyzer=analyzer)
  vectorizer.fit(corpus)
  return vectorizer

def create_features_df(london_corpus, zwickau_corpus, burchard_corpus):
  combined_corpus = london_corpus + zwickau_corpus + burchard_corpus
  vectorizer = create_tf_idf_vectorizer(combined_corpus)

  all_data = create_corpus_features(vectorizer, london_corpus, LONDON_VERSION_LABEL) + create_corpus_features(vectorizer, zwickau_corpus, ZWICKAU_VERSION_LABEL) + create_corpus_features(vectorizer, burchard_corpus, BURCHARD_VERSION_LABEL)

  columns = vectorizer.get_feature_names() + METADATA_COLUMN_NAMES

  return pd.DataFrame(all_data, columns=columns)

def create_X_y(features_df):
  y = features_df['version']
  X = features_df.copy().drop(METADATA_COLUMN_NAMES, axis=1)
  return X, y