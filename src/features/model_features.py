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
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV


LONDON_VERSION_LABEL = 0
ZWICKAU_VERSION_LABEL = 1
BURCHARD_VERSION_LABEL = 2

METADATA_COLUMN_NAMES = ['index', 'corpus_version_label']

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

def create_corpus_features(vectorizer, corpus, n_gram, corpus_label, features):
  data = []

  inner_cosine_similarities = thesisCosineSimilarity.get_inner_version_all_similarities(corpus)[f'{n_gram}_gram'] if INNER_MEAN_COSINE_SIMILARITY_SCORE in features else None

  for index, p in enumerate(corpus):
    p_features = create_p_features(vectorizer, inner_cosine_similarities, corpus, p, index, corpus_label, features)
    data.append(p_features)
  
  return data

def create_tf_idf_vectorizer(corpus, n_gram = 5, analyzer='char'):
  vectorizer = TfidfVectorizer(ngram_range=(n_gram, n_gram), analyzer=analyzer)
  vectorizer.fit(corpus)
  return vectorizer

def create_features_df(
  london_corpus, 
  zwickau_corpus, 
  burchard_corpus, 
  n_gram,
  features = { TFIDF_FEATURE_NAME },
  ):
  # TODO: handle case than corpus is undefined
  combined_corpus = london_corpus + zwickau_corpus # + burchard_corpus

  vectorizer = create_tf_idf_vectorizer(combined_corpus, n_gram) if TFIDF_FEATURE_NAME in features else None

  corpuses = []
  if london_corpus is not None: corpuses.append((london_corpus, LONDON_VERSION_LABEL))
  if zwickau_corpus is not None: corpuses.append((zwickau_corpus, ZWICKAU_VERSION_LABEL))
  if burchard_corpus is not None: corpuses.append((burchard_corpus, BURCHARD_VERSION_LABEL))

  all_features = []
  for corpus, corpus_label in corpuses:
    corpus_features = create_corpus_features(vectorizer, corpus, n_gram, corpus_label, features) 
    all_features = all_features + corpus_features

  # london_features = [] if london_corpus is None else create_corpus_features(vectorizer, london_corpus, n_gram, LONDON_VERSION_LABEL, features) 
  # zwickau_features = [] if zwickau_corpus is None else create_corpus_features(vectorizer, zwickau_corpus, n_gram, ZWICKAU_VERSION_LABEL, features)
  # burchard_features = [] if burchard_corpus is None else create_corpus_features(vectorizer, burchard_corpus, n_gram, BURCHARD_VERSION_LABEL, features)

  # all_features = london_features + zwickau_features + burchard_features

  columns = METADATA_COLUMN_NAMES 
  if TFIDF_FEATURE_NAME in features: columns = columns + vectorizer.get_feature_names()
  if INNER_MEAN_COSINE_SIMILARITY_SCORE in features: columns = columns + [INNER_MEAN_COSINE_SIMILARITY_SCORE]

  return pd.DataFrame(all_features, columns=columns)

def create_X_y(features_df):
  y = features_df['corpus_version_label']
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
    print(f'running: {classifier_name}')
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
  
  return scores_df

def run_grid_search_cv(features_df, classifiers_to_test):
  X, y = create_X_y(features_df)
  X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101,stratify=y)

  # classifiers = [
    
  #   # TODO: MLPClassifier didn't work
  #   ( MLPClassifier(), 'MLPClassifier', { 
  #     'solver': ['lbfgs', 'sgd', 'adam'], 
  #     'alpha': 10.0 ** -np.arange(1, 10), 
  #     'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,10000],
  #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
  #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
  #     'hidden_layer_sizes': [(10,), (20,), (40,), (60,), (80,), (100,), (150,), (200,), (300,), (500,), (700,), (900,), (1500,), (2000,), (50,50,50), (50,100,50)] 
  #     } ), 
  # ]

  

  classifiers = {
    'SVC': ( SVC(), { 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6) / X_train.shape[0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'] } ),
    'DecisionTreeClassifier': (DecisionTreeClassifier(), { 'criterion': ['gini', 'entropy'], 'max_depth': [None,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]} ),
    'GaussianProcessClassifier': ( GaussianProcessClassifier(),  { 'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()] } ),
    'RandomForestClassifier': ( RandomForestClassifier(),  { 
      'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500], 
      'criterion': ['gini', 'entropy', 'log_loss'], 
      'max_features': ['auto', 'sqrt', 'log2'], 
      'max_depth' : [4,5,6,7,8,9,10,11,12], 
      'random_state': [0], 
      'max_features': ['auto', 'sqrt', 'log2']
      }),
  'GaussianNB': ( GaussianNB(), { 'var_smoothing': np.logspace(0,-9, num=100) } ),
  'KNeighborsClassifier': ( KNeighborsClassifier(),  {
      'n_neighbors': [3,5,11,19],
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
      }),
  'AdaBoostClassifier': ( AdaBoostClassifier(), { 'n_estimators':[5, 10, 30, 50, 100, 300, 500,1000,2000], 'learning_rate':[.001,0.01,.1, .5, 1, 2] } ),
  'QuadraticDiscriminantAnalysis': ( QuadraticDiscriminantAnalysis(), { 
      'reg_param': [0.00001, 0.0001, 0.001,0.01, 0.1, 0.2, 0.3, 0.4, 0.5], 
      'store_covariance': [True, False],
      'tol': (0.0001, 0.001,0.01, 0.1)
      })
  }

  if classifiers_to_test is None: classifiers_to_test = list(classifiers.keys())
  print(f'testing classifiers: {classifiers_to_test}')

  scores_df = pd.DataFrame(dtype=float)
  grid_results = []

  for classifier_name in classifiers_to_test:
    classifier, classifiers_grid_search_params = classifiers[classifier_name]
    print(f'running: {classifier_name}')
    param_grid = GridSearchCV(
      classifier, 
      classifiers_grid_search_params, 
      return_train_score = True, 
      cv = 10, 
      n_jobs = -1,
    )
    param_grid.fit(X_train, y_train)
    grid_results.append(param_grid)

  # for classifier, classifier_name, classifiers_grid_search_params in classifiers:
    # print(f'running: {classifier_name}')
    # param_grid = GridSearchCV(
    #   classifier, 
    #   classifiers_grid_search_params, 
    #   return_train_score = True, 
    #   cv = 10, 
    #   n_jobs = -1,
    # )
    # param_grid.fit(X_train, y_train)
    # grid_results.append(param_grid)

    # scores_df.loc[classifier_name, 'best_score_'] = param_grid.best_score_
    # scores_df.loc[classifier_name, 'best_estimator_'] = param_grid.best_estimator_
  
  return scores_df, grid_results