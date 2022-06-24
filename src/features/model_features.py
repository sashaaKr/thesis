from statistics import mean
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import similarities.cosine as thesisCosineSimilarity
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

LONDON_VERSION_LABEL = 0
ZWICKAU_VERSION_LABEL = 1
BURCHARD_VERSION_LABEL = 2

METADATA_COLUMN_NAMES = ['index', 'version']

TFIDF_FEATURE_NAME = 'tfidf'
INNER_MEAN_COSINE_SIMILARITY_SCORE = 'inner_mean_cosine_similarity_score'

def create_tfidf_features(vectorizer, p):
  p_features_vectorizer = vectorizer.transform([p]).toarray()[0]
  return p_features_vectorizer

# def create_inner_similarity_score(corpus, p):


def create_p_features(vectorizer, inner_cosine_similarities, corpus, p, p_index, corpus_label, features):
  all_features = np.array([p_index, corpus_label])

  if TFIDF_FEATURE_NAME in features: 
    all_features = np.append(all_features, create_tfidf_features(vectorizer, p))

  if INNER_MEAN_COSINE_SIMILARITY_SCORE in features:
    mean_cosine_similarity_score = np.mean([i[1] for i in inner_cosine_similarities[p_index]])
    all_features = np.append(all_features, [mean_cosine_similarity_score])

  return all_features

def create_corpus_features(vectorizer, corpus, corpus_label, features):
  data = []

  inner_cosine_similarities = thesisCosineSimilarity.get_inner_version_all_similarities(corpus)['5_gram'] if INNER_MEAN_COSINE_SIMILARITY_SCORE in features else None

  for index, p in enumerate(corpus):
    p_features = create_p_features(vectorizer, inner_cosine_similarities, corpus, p, index, corpus_label, features)
    data.append(p_features)
  
  return data

def create_tf_idf_vectorizer(corpus, n_gram = 5, analyzer='char'):
  vectorizer = TfidfVectorizer(ngram_range=(n_gram, n_gram), analyzer=analyzer)
  vectorizer.fit(corpus)
  return vectorizer

def create_features_df(london_corpus, zwickau_corpus, burchard_corpus, features = { TFIDF_FEATURE_NAME }):
  # TODO: handle case than corpus is undefined
  combined_corpus = london_corpus + zwickau_corpus # + burchard_corpus

  vectorizer = create_tf_idf_vectorizer(combined_corpus) if TFIDF_FEATURE_NAME in features else None

  london_features = [] if london_corpus is None else create_corpus_features(vectorizer, london_corpus, LONDON_VERSION_LABEL, features) 
  zwickau_features = [] if zwickau_corpus is None else create_corpus_features(vectorizer, zwickau_corpus, ZWICKAU_VERSION_LABEL, features)
  burchard_features = [] if burchard_corpus is None else create_corpus_features(vectorizer, burchard_corpus, BURCHARD_VERSION_LABEL, features)

  all_features = london_features + zwickau_features + burchard_features

  columns = METADATA_COLUMN_NAMES 
  if TFIDF_FEATURE_NAME in features: columns = columns + vectorizer.get_feature_names()
  if INNER_MEAN_COSINE_SIMILARITY_SCORE in features: columns = columns + [INNER_MEAN_COSINE_SIMILARITY_SCORE]

  return pd.DataFrame(all_features, columns=columns)

def create_X_y(features_df):
  y = features_df['version']
  X = features_df.copy().drop(METADATA_COLUMN_NAMES, axis=1)
  return X, y

def run_models(features_df):
  X, y = create_X_y(features_df)
  X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101,stratify=y)

  scores_df = pd.DataFrame(dtype=float)
  scoring = ('precision_macro', 'recall_macro', 'f1_macro', 'f1_micro', 'f1_weighted', 'accuracy')

  classifiers = [
    ( SVC(kernel="linear", C=0.025), 'SVM_linear' ),
    ( SVC(gamma=2, C=1), 'SVM_RBF' ),
    ( DecisionTreeClassifier(), 'DecisionTreeClassifier' ),
    ( GaussianProcessClassifier(1.0 * RBF(1.0)), 'GaussianProcessClassifier' ),
    ( RandomForestClassifier(), 'RandomForestClassifier' ),
    ( MLPClassifier(alpha=1, max_iter=1000), 'MLPClassifier' ),
    ( GaussianNB(), 'GaussianNB' ),
    ( KNeighborsClassifier(), 'KNeighborsClassifier' ),
    ( AdaBoostClassifier(), 'AdaBoostClassifier' ),
    ( QuadraticDiscriminantAnalysis(), 'QuadraticDiscriminantAnalysis')
  ]

  for classifier, classifier_name in classifiers:
    classifier_cross_validate_score = cross_validate(
      classifier,
      X_train,
      y_train,
      cv=10,
      scoring=scoring
      # n_jobs = -1
    )
    
    for s in scoring:
      scores_df.loc[classifier_name, s] = classifier_cross_validate_score[f'test_{s}'].mean()

  # svm_linear_cross_validate_score = cross_validate(
  #   SVC(kernel="linear", C=0.025),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  #   # n_jobs = -1
  # )

  # for s in scoring:
  #   scores_df.loc['SVM_linear', s] = svm_linear_cross_validate_score[f'test_{s}'].mean()

  # rbf_svm_cross_validate_score = cross_validate(
  #   SVC(gamma=2, C=1),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  #   # n_jobs = -1
  # )

  # for s in scoring:
  #   scores_df.loc['SVM_RBF', s] = rbf_svm_cross_validate_score[f'test_{s}'].mean()
  
  # decision_tree_validate_score = cross_validate(
  #   DecisionTreeClassifier(),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  #   # n_jobs = -1
  # )

  # for s in scoring:
  #   scores_df.loc['DecisionTreeClassifier', s] = decision_tree_validate_score[f'test_{s}'].mean()

  # gaussian_process_classifier_score = cross_validate(
  #   GaussianProcessClassifier(1.0 * RBF(1.0)),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  # )

  # for s in scoring:
  #   scores_df.loc['GaussianProcessClassifier', s] = gaussian_process_classifier_score[f'test_{s}'].mean()

  # random_forest_cross_validate = cross_validate(
  #   RandomForestClassifier(),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  # )

  # for s in scoring:
  #   scores_df.loc['RandomForestClassifier', s] = random_forest_cross_validate[f'test_{s}'].mean()

  # mlp_cross_validate = cross_validate(
  #   MLPClassifier(alpha=1, max_iter=1000),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  # )

  # gaussian_nb_cross_validate = cross_validate(
  #   GaussianNB(),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  # )

  # for s in scoring:
  #   scores_df.loc['GaussianNB', s] = gaussian_nb_cross_validate[f'test_{s}'].mean()
  
  # k_neighbors_classifier_cross_validate = cross_validate(
  #   KNeighborsClassifier(),
  #   X_train,
  #   y_train,
  #   cv=10,
  #   scoring=scoring
  # )

  # for s in scoring:
  #   scores_df.loc['KNeighborsClassifier', s] = k_neighbors_classifier_cross_validate[f'test_{s}'].mean()
  
  return scores_df