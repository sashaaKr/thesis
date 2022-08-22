import pickle
from statistics import mean
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import similarities.cosine as thesisCosineSimilarity
import features.tf_idf.n_gram as thesisNgram
import data.reader as thesisDataReader
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


LONDON_VERSION_LABEL = 0
ZWICKAU_VERSION_LABEL = 1
BURCHARD_VERSION_LABEL = 2

METADATA_COLUMN_NAMES = ['index', 'corpus_version_label']

TFIDF_FEATURE_NAME = 'tfidf'
INNER_MEAN_COSINE_SIMILARITY_SCORE = 'inner_mean_cosine_similarity_score'

def create_tfidf_features(vectorizer, p):
  p_features_vectorizer = vectorizer.transform([p]).toarray()[0]
  return p_features_vectorizer

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

  n_gram_feature_name = f'{n_gram}_gram' if type(n_gram) is int else f'{n_gram[0]}_{n_gram[1]}_gram'
  print(f'n_gram_feature_name: {n_gram_feature_name}')
  inner_cosine_similarities = thesisCosineSimilarity.get_inner_version_all_similarities(corpus)[n_gram_feature_name] if INNER_MEAN_COSINE_SIMILARITY_SCORE in features else None

  for index, p in enumerate(corpus):
    p_features = create_p_features(vectorizer, inner_cosine_similarities, corpus, p, index, corpus_label, features)
    data.append(p_features)
  
  return data

def create_tf_idf_vectorizer(corpus, n_gram = 5, analyzer='char'):
  vectorizer = TfidfVectorizer(ngram_range=thesisNgram.get_n_gram_range(n_gram), analyzer=analyzer)
  vectorizer.fit(corpus)
  return vectorizer

def create_and_save_london_zwickau_vectorizer(*, london_corpus, zwickau_corpus, name, n_gram = (2,5), analyzer = 'char'):
  combined_corpus = london_corpus + zwickau_corpus
  vectorizer = create_tf_idf_vectorizer(combined_corpus, n_gram, analyzer)
  
  with open(f'../computed_data/models/london_vs_zwickau/vectorizers/{name}.pk','wb') as f:
    pickle.dump(vectorizer, f)

def create_and_save_london_burchard_vectorizer(*, london_corpus, burchard_corpus, name, n_gram = (2,5), analyzer = 'char'):
  combined_corpus = london_corpus + burchard_corpus
  vectorizer = create_tf_idf_vectorizer(combined_corpus, n_gram, analyzer)
  
  with open(f'../computed_data/models/london_vs_burchard/vectorizers/{name}.pk','wb') as f:
    pickle.dump(vectorizer, f)

def create_and_save_zwickau_burchard_vectorizer(*, zwickau_corpus, burchard_corpus, name, n_gram = (2,5), analyzer = 'char'):
  combined_corpus = zwickau_corpus + burchard_corpus
  vectorizer = create_tf_idf_vectorizer(combined_corpus, n_gram, analyzer)
  
  with open(f'../computed_data/models/zwickau_vs_burchard/vectorizers/{name}.pk','wb') as f:
    pickle.dump(vectorizer, f)

def load_london_zwickau_vectorizer(name):
  with open(f'../computed_data/models/london_vs_zwickau/vectorizers/{name}.pk','rb') as f:
    return pickle.load(f)

def load_london_burchard_vectorizer(name):
  with open(f'../computed_data/models/london_vs_burchard/vectorizers/{name}.pk','rb') as f:
    return pickle.load(f)

def load_zwickau_burchard_vectorizer(name):
  with open(f'../computed_data/models/zwickau_vs_burchard/vectorizers/{name}.pk','rb') as f:
    return pickle.load(f)

def create_features_df(
  london_corpus, 
  zwickau_corpus, 
  burchard_corpus, 
  n_gram,
  features = { TFIDF_FEATURE_NAME },
  analyzer = 'char',
  vectorizer = None
  ):
  combined_corpus = []
  corpuses = []

  if london_corpus is not None: 
    corpuses.append((london_corpus, LONDON_VERSION_LABEL))
    combined_corpus = combined_corpus + london_corpus
  if zwickau_corpus is not None: 
    corpuses.append((zwickau_corpus, ZWICKAU_VERSION_LABEL))
    combined_corpus = combined_corpus + zwickau_corpus
  if burchard_corpus is not None: 
    corpuses.append((burchard_corpus, BURCHARD_VERSION_LABEL))
    combined_corpus = combined_corpus + burchard_corpus

  if vectorizer is None:
    vectorizer = create_tf_idf_vectorizer(combined_corpus, n_gram, analyzer) if TFIDF_FEATURE_NAME in features else None

  all_features = []
  for corpus, corpus_label in corpuses:
    corpus_features = create_corpus_features(vectorizer, corpus, n_gram, corpus_label, features) 
    all_features = all_features + corpus_features

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

  # TODO: add LinearRegression model
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
    ( xgb.XGBClassifier(), 'XGBClassifier' ),
    # ( xgb.XGBRFClassifier(), 'XGBRFClassifier' ),
    # ( QuadraticDiscriminantAnalysis(), 'QuadraticDiscriminantAnalysis')
  ]

  cross_validate_results = []
  for classifier, classifier_name in classifiers:
    print(f'running: {classifier_name}')
    classifier_cross_validate_score = cross_validate(
      classifier,
      X,
      LabelEncoder().fit_transform(y) if classifier_name == 'XGBClassifier' else y,
      cv=10,
      scoring=scoring
    )

    cross_validate_results.append(classifier_cross_validate_score)
    for s in scoring:
      scores_df.loc[classifier_name, s] = classifier_cross_validate_score[f'test_{s}'].mean()
  
  return scores_df, cross_validate_results

def run_grid_search_cv(features_df, classifiers_to_test):
  X, y = create_X_y(features_df)
  # X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101,stratify=y)

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
    'SVC': ( SVC(), { 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6) / X.shape[0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'] } ),
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
      }),

      'XGBClassifier': ( xgb.XGBClassifier(), {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2),
        'gamma':[i/10.0 for i in range(0,5)],
        # 'subsample':[i/10.0 for i in range(6,10)],
        # 'colsample_bytree':[i/10.0 for i in range(6,10)],
        # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
      }),

      'XGBRFClassifier': ( xgb.XGBRFClassifier(), {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2),
        'gamma':[i/10.0 for i in range(0,5)],
        # 'subsample':[i/10.0 for i in range(6,10)],
        # 'colsample_bytree':[i/10.0 for i in range(6,10)],
        # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
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

    y = LabelEncoder().fit_transform(y) if classifier_name == 'XGBClassifier' or classifier_name == 'XGBRFClassifier' else y
    param_grid.fit(X, y)
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

def get_model_wrong_prediction(*, features_df, classifier,  splits = 10):
  results = []
  X, y = create_X_y(features_df)

  # TODO: test with shuffle
  skf = StratifiedKFold(n_splits = splits) # TODO: add random_state

  for train_indexes, test_indexes in skf.split(X, y):
    result = []

    X_train, X_test = X.iloc[train_indexes], X.iloc[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    for (prediction, label, index) in zip(predicted, y_test, test_indexes):
      if prediction != label:
        index_in_text =  features_df.loc[index, 'index']
        print('Row', index_in_text, 'has been classified as ', prediction, 'and should be ', label)
        result.append((prediction, label, index_in_text))
    
    results.append(result)
    print(f'score is: {classifier.score(X_test, y_test)}')
  
  return results

def get_london_vs_zwickau_best_model():
  with open('../computed_data/models/london_vs_zwickau/best_models/AdaBoostClassifier(learning_rate=0.01, n_estimators=1000)_0_799.pkl', 'rb') as f:
    clf = pickle.load(f)
  
  return clf

def get_london_vs_burchard_best_model():
  with open('../computed_data/models/london_vs_burchard/best_models/RandomForestClassifier(max_depth=9, n_estimators=200, random_state=0)_0_781_tfidf_2_5_gram_cosine_similarity_long_p.pkl', 'rb') as f:
    clf = pickle.load(f)
  
  return clf

def get_zwickau_vs_burchard_best_model():
  with open('../computed_data/models/zwickau_vs_burchard/best_models/AdaBoostClassifier(learning_rate=1, n_estimators=2000)_0_912_tfidf_2_5_gram_cosine_similarity_long_p.pkl', 'rb') as f:
    return pickle.load(f)

def save_london_vs_zwickau_best_model(clf, name):
  with open(f'../computed_data/models/best_models/london_vs_zwickau/{name}','wb') as f:
    pickle.dump(clf, f)

def save_zwickau_vs_burchard_best_model(clf, name):
  with open(f'../computed_data/models/zwickau_vs_burchard/best_models/{name}.pkl','wb') as f:
    pickle.dump(clf, f)

def save_london_vs_burchard_best_model(clf, name):
  with open(f'../computed_data/models/london_vs_burchard/best_models/{name}.pkl','wb') as f:
    pickle.dump(clf, f)

def create_burchard_features_tfidf_2_5_gram_cosine_similarity_long_p_df(vectorizer = None):
  return create_features_df(
    None,
    None,
    thesisDataReader.get_burchard_candidate_version_based_on_strongly_similar_london_base_long_p(),
    n_gram = (2,5),
    features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
    vectorizer = vectorizer
  )

def create_london_features_tfidf_2_5_gram_cosine_similarity_long_p_df(vectorizer = None):
  return create_features_df(
    thesisDataReader.get_london_poorly_similar_with_chops_corpus_without_word_processing_long_p(),
    None,
    None,
    n_gram = (2,5),
    features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
    vectorizer = vectorizer
  )

def create_zwickau_features_tfidf_2_5_gram_cosine_similarity_long_p_df(vectorizer = None):
  return create_features_df(
    None,
    thesisDataReader.get_zwickau_poorly_similar_with_chops_corpus_without_word_processing_long_p(),
    None,
    n_gram = (2,5),
    features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
    vectorizer = vectorizer
  ) 

def version_label_to_human_readable(version_label):
  if version_label == LONDON_VERSION_LABEL:
    return 'London'
  
  if version_label == ZWICKAU_VERSION_LABEL:
    return 'Zwickau'

  if version_label == BURCHARD_VERSION_LABEL:
    return 'Burchard'
  
  raise 'Unknown version label'

def run_london_zwickau_best_model_on_burchard_wrong_predictions(wrong_predictions):
  __wrong_prediction = wrong_predictions.copy()

  london_vs_zwickau_vectorizer = load_london_zwickau_vectorizer('features_tfidf_2_5_gram_cosine_similarity_long_p')
  london_vs_zwickau_best_model = get_london_vs_zwickau_best_model()

  burchard_features_tfidf_2_5_gram_cosine_similarity_long_p_df = create_burchard_features_tfidf_2_5_gram_cosine_similarity_long_p_df(london_vs_zwickau_vectorizer)
  X, y = create_X_y(burchard_features_tfidf_2_5_gram_cosine_similarity_long_p_df)

  for index, wrong_prediction in enumerate(__wrong_prediction):
    is_wrong_burchard = wrong_prediction[3]

    fixed_prediction = '--'
    if not is_wrong_burchard:
      is_london_or_zwickau = london_vs_zwickau_best_model.predict([X.loc[index]])[0]
      print(f'predicted: {version_label_to_human_readable(is_london_or_zwickau)}')
      fixed_prediction = version_label_to_human_readable(is_london_or_zwickau)
    
    __wrong_prediction[index].append(fixed_prediction)

  for index, wrong_prediction in enumerate(__wrong_prediction):
    is_wrong_burchard = wrong_prediction[4]

    fixed_prediction = '--'
    if not is_wrong_burchard:
      is_london_or_zwickau = london_vs_zwickau_best_model.predict([X.loc[index]])[0]
      print(f'predicted: {version_label_to_human_readable(is_london_or_zwickau)}')
      fixed_prediction = version_label_to_human_readable(is_london_or_zwickau)
    
    __wrong_prediction[index].append(fixed_prediction)

  return __wrong_prediction

def run_london_burchard_best_model_on_london_wrong_predictions(wrong_predictions):
  __wrong_prediction = wrong_predictions.copy()
  
  london_burchard_vectorizer = load_london_burchard_vectorizer('features_tfidf_2_5_gram_cosine_similarity_long_p')
  london_burchard_best_model = get_london_vs_burchard_best_model()

  london_features_tfidf_2_5_gram_cosine_similarity_long_p_df = create_london_features_tfidf_2_5_gram_cosine_similarity_long_p_df(london_burchard_vectorizer)
  X, y = create_X_y(london_features_tfidf_2_5_gram_cosine_similarity_long_p_df)

  for index, wrong_prediction in enumerate(__wrong_prediction):
    is_wrong_london = wrong_prediction[1]

    fixed_prediction = '--'
    if not is_wrong_london:
      is_london_or_burchard = london_burchard_best_model.predict([X.loc[index]])[0]
      print(f'predicted: {version_label_to_human_readable(is_london_or_burchard)}')
      fixed_prediction = version_label_to_human_readable(is_london_or_burchard)
    
    __wrong_prediction[index].append(fixed_prediction)
  
  return __wrong_prediction

def run_zwickau_burchard_best_model_on_zwickau_wrong_predictions(wrong_predictions):
  __wrong_prediction = wrong_predictions.copy()

  zwickau_burchard_vectorizer = load_zwickau_burchard_vectorizer('features_tfidf_2_5_gram_cosine_similarity_long_p')
  zwickau_burchard_best_model = get_zwickau_vs_burchard_best_model()

  zwickau_features_tfidf_2_5_gram_cosine_similarity_long_p_df = create_zwickau_features_tfidf_2_5_gram_cosine_similarity_long_p_df(zwickau_burchard_vectorizer)
  X, y = create_X_y(zwickau_features_tfidf_2_5_gram_cosine_similarity_long_p_df)

  for index, wrong_prediction in enumerate(__wrong_prediction):
    is_wrong_london = wrong_prediction[1]

    fixed_prediction = '--'
    if not is_wrong_london:
      is_london_or_burchard = zwickau_burchard_best_model.predict([X.loc[index]])[0]
      print(f'predicted: {version_label_to_human_readable(is_london_or_burchard)}')
      fixed_prediction = version_label_to_human_readable(is_london_or_burchard)
    
    __wrong_prediction[index].append(fixed_prediction)

  return __wrong_prediction

def create_london_zwickau_with_processing_features_tfidf_2_5_gram_cosine_similarity_long_p_df():
  return create_features_df(
    thesisDataReader.get_london_poorly_similar_with_chops_long_p_corpus(),
    thesisDataReader.get_zwickau_poorly_similar_with_chops_long_p_corpus(),
    None,
    n_gram = (2,5),
    features = { 'tfidf', 'inner_mean_cosine_similarity_score' }
    )

def create_london_burchard_with_processing_features_tfidf_2_5_gram_cosine_similarity_long_p_df():
  return create_features_df(
    thesisDataReader.get_london_poorly_similar_with_chops_long_p_corpus(),
    None,
    thesisDataReader.get_burchard_candidate_version_based_on_strongly_similar_london_base_long_p_corpus(),
    n_gram = (2,5),
    features = { 'tfidf', 'inner_mean_cosine_similarity_score' }
    )

def create_zwickau_burchard_with_processing_features_tfidf_2_5_gram_cosine_similarity_long_p_df():
  return create_features_df(
    None,
    thesisDataReader.get_zwickau_poorly_similar_with_chops_long_p_corpus(),
    thesisDataReader.get_burchard_candidate_version_based_on_strongly_similar_london_base_long_p_corpus(),
    n_gram = (2,5),
    features = { 'tfidf', 'inner_mean_cosine_similarity_score' }
    )

