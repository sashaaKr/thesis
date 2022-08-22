import os
import pickle
import numpy as np
import pandas as pd
import data.reader as thesisDataReader
import features.tf_idf.n_gram as thesisTfIdfNgramFeatures
import utils.utils as thesisUtils
import features.count_vectorizer.n_gram as thesisCountVectorizerNgramFeatures
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import matplotlib.pyplot as plt

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.join(USER_HOME_DIR, 'thesis') 

# FEATURE_NAMES = {
#     # TF_IDF: {},
#     COUNT_VECTORIZER: {
#         '2_gram': '2_gram',
#         '3_gram': '3_gram',
#         '4_gram': '4_gram',
#         '5_gram': '5_gram',
#     }
# }

FEATURES = [
    # ['2_gram', thesisTfIdfNgramFeatures.create_2_gram],
    # ['3_gram', thesisTfIdfNgramFeatures.create_3_gram],
    # ['4_gram', thesisTfIdfNgramFeatures.create_4_gram],
    ['5_gram', thesisTfIdfNgramFeatures.create_5_gram],
    # ['6_gram', thesisTfIdfNgramFeatures.create_6_gram],
    # ['7_gram', thesisTfIdfNgramFeatures.create_7_gram],
    # ['8_gram', thesisTfIdfNgramFeatures.create_8_gram],

    ['2_5_gram', thesisTfIdfNgramFeatures.create_2_5_gram],
    # ['3_5_gram', thesisTfIdfNgramFeatures.create_3_5_gram],

    # count_vectorizer_features
    # ['count_vectorizer_5_gram', thesisCountVectorizerNgramFeatures.create_5_gram]
]

def calculate_p_to_version_similarity(paragraph_to_test, version_corpus, get_features):
    mixed_corpus = [paragraph_to_test] + version_corpus
    mixed_features_df = get_features(mixed_corpus)
    mixed_similarities = cosine_similarity(mixed_features_df, mixed_features_df)
    mixed_similarities_df = pd.DataFrame(mixed_similarities)
    
    relevant_data = mixed_similarities_df.iloc[0,:]
    return relevant_data

def get_ordered_similarities_without_self(series_similarity):
    without_self_p = series_similarity.drop(index=[0])
    reindexed = without_self_p.set_axis(range(0, without_self_p.size))
    sorted_similarities = reindexed.sort_values(ascending=False)
    # without_self_p = sorted_similatieis.drop(index=[0])
    return list(sorted_similarities.items())

def get_inner_version_best_similarities(version):
    all_best_result = {}

    for [feature_name, get_feature] in FEATURES:
        version_df = get_feature(version)
        similarities = cosine_similarity(version_df, version_df)
        similarities_df = pd.DataFrame(similarities)
        
        all_best = []
        for index, row in similarities_df.iterrows():
            without_self = row.drop(index=[index])
            sorted_similatieis = without_self.sort_values(ascending=False)
            sorted_similarities_list = list(sorted_similatieis.items())
            all_best.append(sorted_similarities_list[0])
        
        all_best_result[feature_name] = all_best
    return all_best_result
    
def get_cross_version_best_similarities(version_1, version_2, features=FEATURES):
    all_best_result = []
    
    for i, p in enumerate(version_1):
        all_best = {}
        uniq_best = set()

        for [featire_name, get_feature] in features:
            smltr = calculate_p_to_version_similarity(p, version_2, get_feature)
            smltr_ordered = get_ordered_similarities_without_self(smltr)
            
            all_best[featire_name] = smltr_ordered[0]
            uniq_best.add(smltr_ordered[0][0])

        all_best_result.append(all_best)
        
    return all_best_result

def get_cross_version_5_gram_best_similarities(verstion_1, version_2):
    features = [['5_gram', thesisTfIdfNgramFeatures.create_5_gram]]
    return get_cross_version_best_similarities(verstion_1, version_2, features)

def create_version_to_version_5_gram_comparison_csv(
    file_name, 
    version_1_corpus, 
    version_1_name, 
    version_2_corpus,
    version_2_name,
    version_1_section_indexes,
    version_2_section_indexes
    ):
    all_best_result = get_cross_version_best_similarities(version_1_corpus, version_2_corpus)

    data_frame_data = []
    for i, d in enumerate(all_best_result):
        n_gram_5 = d['5_gram']

        similarity_score = n_gram_5[1]
        paragraph = n_gram_5[0]
        
        version_1_text = version_1_corpus[i]
        version_2_text = version_2_corpus[paragraph]
        
        data = [
            paragraph, 
            similarity_score, 
            version_1_text, 
            version_2_text,
            int(version_1_section_indexes[i]),
            int(version_2_section_indexes[paragraph]),
            i - paragraph,
            int(version_1_section_indexes[i]) == int(version_2_section_indexes[paragraph])

    ]
        data_frame_data.append(data)

    data_frame_cols = [
        f'{version_2_name}_p_#', 
        'similarity_score', 
        version_1_name, 
        version_2_name,
        f'{version_1_name}_section_index',
        f'{version_2_name}_section_index',
        'shift_in_paragraphs',
        'same_section_similarity'
        ]

    text_to_text_df = pd.DataFrame(data=np.array(data_frame_data), columns=data_frame_cols)
    text_to_text_df.index.rename(f'{version_1_name}_p_#', inplace=True)
    text_to_text_df.to_csv(f'../computed_data/text_to_text/{file_name}.csv')
    return text_to_text_df

def london_to_zwickau_best_similarities():
    london_corpus = thesisDataReader.get_london_corpus()
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    return get_cross_version_best_similarities(london_corpus, zwickau_corpus)

def zwickau_to_london_best_similarities():
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    london_corpus = thesisDataReader.get_london_corpus()
    return get_cross_version_best_similarities(zwickau_corpus, london_corpus)

def get_london_with_self_best_similarities():
    london_corpus = thesisDataReader.get_london_corpus()
    return get_inner_version_best_similarities(london_corpus)

def get_zwickau_with_self_best_similarities():
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    return get_inner_version_best_similarities(zwickau_corpus)

def get_inner_version_all_similarities(version):
    all_results = {}

    for [feature_name, get_feature] in FEATURES:
        version_df = get_feature(version)
        similarities = cosine_similarity(version_df, version_df)
        similarities_df = pd.DataFrame(similarities)

        results = []
        for index, row in similarities_df.iterrows():
            without_self = row.drop(index=[index])
            sorted_similatieis = without_self.sort_values(ascending=False)
            sorted_similarities_list = list(sorted_similatieis.items())
            results.append(sorted_similarities_list)
        
        all_results[feature_name] = results
    return all_results

def get_zwickau_with_self_all_similarities():
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    return get_inner_version_all_similarities(zwickau_corpus)

def get_london_with_self_all_similarities():
    london_corpus = thesisDataReader.get_london_corpus()
    return get_inner_version_all_similarities(london_corpus)

def get_p_stats(version, p_index, p_similarities, feature_name, cross_inner, compared_to_version):
    data = [feature_name, p_index, cross_inner]
    smlrts = list(map(lambda r: r[1], p_similarities))
            
    df_describe = pd.DataFrame(np.array(smlrts))
    describe_result = df_describe.describe()

    data.append(describe_result.at['mean', 0])
    data.append(describe_result.at['std', 0])
    data.append(describe_result.at['min', 0])
    data.append(describe_result.at['25%', 0])
    data.append(describe_result.at['50%', 0])
    data.append(describe_result.at['75%', 0])
    data.append(describe_result.at['max', 0])

    value_counts = df_describe.value_counts()
    if 0 in value_counts:
        data.append(value_counts[0])
    else:
        data.append(0)
            
    data.append(len(version[p_index]))

    most_similar = p_similarities[0]
    most_similar_index = most_similar[0]
    most_similar_similarity = most_similar[1]
    data.append(most_similar_index)
    data.append(most_similar_similarity)
    data.append(len(compared_to_version[most_similar_index]))
    
    return data

def create_statistics_df(version_1, version_2, version_1_name):
    all_data = []
    columns = [
        'feature_name',
        'p_#',
        'cross/inner',
        'mean',
        'std',
        'min',
        '25%',
        '50%',
        '75%',
        'max',
        '# of 0 similarities',
        'p_length',

        'most_similar_p_#',
        'most_similar_score',
        'most_similar_p_length',

        'most_similar_dropped',
        'most_similar_dropped_p_#',
        'most_similar_dropperd_score',
        'most_similar_dropped_p_length',
        'version'
    ]
    
    all_similarities = get_inner_version_all_similarities(version_1)
    for feature_name in all_similarities:
        for p_index, p_similarities in enumerate(all_similarities[feature_name]):
            data = get_p_stats(
                version_1,
                p_index,
                p_similarities,
                feature_name,
                'inner',
                version_1
            )
            # data of most similart cross version
            # not relevant then compared inner data
            most_similar_dropped = None
            closest_index = None
            closest_similarity = None
            closest_p_length = None

            data.append(most_similar_dropped)
            data.append(closest_index)
            data.append(closest_similarity)
            data.append(closest_p_length)

            all_data.append(data)
            
    
    for [feature_name, get_feature] in FEATURES:
        for p_index, p in enumerate(version_1):
            smltr = calculate_p_to_version_similarity(p, version_2, get_feature)
            smltr_ordered = get_ordered_similarities_without_self(smltr)
            
            closest = smltr_ordered[0]
            closest_similarity = closest[1]
            most_similar_dropped = False
            if closest_similarity > 0.6:
                smltr_ordered.pop(0)
                most_similar_dropped = True

            p_similarities = list(map(lambda r: r[1], smltr_ordered))
            results = get_p_stats(
                version_1,
                p_index,
                smltr_ordered,
                feature_name,
                'cross',
                version_2
            )
            results.append(most_similar_dropped)

            if most_similar_dropped:
                closest_index = closest[0]
                closest_similarity = closest[1]
                closest_p_length = len(version_2[closest_index])
                results.append(closest_index)
                results.append(closest_similarity)
                results.append(closest_p_length)
            else:
                results.append(None)
                results.append(None)
                results.append(None)

            all_data.append(results)
    
    for d in all_data:
        d.append(version_1_name)
            
    return pd.DataFrame(all_data, columns=columns)

# the difference between this impelementation and in CrossVersionSimilarity is that here self not removed
def cross_version_similarity(version_1_corpus, version_2_corpus, get_features):
  res = []

  for i, p in enumerate(version_1_corpus):
    temp_corpus = [p] + version_2_corpus
    df_features = get_features(temp_corpus)
    temp_similarities = cosine_similarity(df_features, df_features)
    res.append(temp_similarities[0])

  return res

def cross_version_similarity_5_gram(version_1_corpus, version_2_corpus):
  return cross_version_similarity(version_1_corpus, version_2_corpus, thesisTfIdfNgramFeatures.create_5_gram)

def get_max_similarity_per_p(similarities):
  res = []

  for index, value in enumerate(similarities):
    max_indices = thesisUtils.get_n_indexes_of_max_values(value, 6)
    max_indices_without_self = max_indices[:-1]
    max_similarity = value[max_indices_without_self[-1]]
    res.append(max_similarity)

  return res

class SimilarityMatch:
  def __init__(self, original_index, match_index, score, original_text, match_text):
    self.score = score
    self.match_index = match_index
    self.original_index = original_index
    self.original_text = original_text
    self.match_text = match_text
  
  def __repr__(self) -> str:
    return f'{self.original_index} -> {self.match_index}: {self.score}'

class SimilarityMatchList:
  def __init__(self):
    self.matches = []
  
  def __len__(self):
    return len(self.matches)
  
  def __iter__(self):
    for match in self.matches:
      yield match

  def __getitem__(self, index):
    return self.matches[index]
  
  def __repr__(self) -> str:
    return '\n'.join(map(str, self.matches))
  
  def append(self, match):
    self.matches.append(match)
  
  def original_indexes(self):
    return [i.original_index for i in self.matches]
  
  def scores(self):
    return [i.score for i in self.matches]

class CrossVersionSimilarity:
  def __init__(self, corpus_1, corpus_2, vectorizer):
    self.corpus_1 = corpus_1
    self.corpus_2 = corpus_2
    self.vectorizer = vectorizer

    # self.raw_matches = []
    # self.best_matches = SimilarityMatchList()
    # self.all_matches_without_self = []

  def load(self):
    path = self.__path_to_data_store()
    data_to_load = ['raw_matches', 'best_matches', 'all_matches_without_self']

    for data in data_to_load:
      with open(os.path.join(path, f'{data}.pickle'), 'rb') as f:
        setattr(self, data, pickle.load(f))

    with open(os.path.join(path, 'raw_matches.pickle'), 'rb') as f:
      raw_matches = pickle.load(f)
      self.raw_matches = raw_matches
    
  def save(self):
    path = self.__path_to_data_store()
    data_to_load = ['raw_matches', 'best_matches', 'all_matches_without_self']

    for data in data_to_load:
      with open(os.path.join(path, f'{data}.pickle'), 'wb') as f:
        pickle.dump(getattr(self, data), f, pickle.HIGHEST_PROTOCOL)

  def __path_to_data_store(self):
    vectorizer_name = str(self.vectorizer.name).replace(" ", "")
    return os.path.join(
      ROOT, 
      'src',
      'similarities', 
      'saved_similarities', 
      f'{self.corpus_1.name}_{self.corpus_2.name}_{vectorizer_name}'
      )

  def calculate(self):
    self.raw_matches = []
    self.best_matches = SimilarityMatchList()
    self.all_matches_without_self = []
    
    for i, p in enumerate(self.corpus_1.corpus):
      smltr = calculate_p_to_version_similarity(p, self.corpus_2.corpus, self.vectorizer.get_features)
      smltr_ordered = get_ordered_similarities_without_self(smltr)

      self.raw_matches.append(smltr.values.tolist())
      best_match = SimilarityMatch(i, smltr_ordered[0][0], smltr_ordered[0][1], p, self.corpus_2.corpus[smltr_ordered[0][0]])
      self.best_matches.append(best_match)
      
      self.all_matches_without_self.append(smltr_ordered)

  def text_alignment(self):
    alignment = {}
    for match in self.best_matches:
      text_1 = self.corpus_1.corpus[match.original_index]
      text_2 = self.corpus_2.corpus[match.match_index]
      alignment[text_1] = text_2
    return alignment
  
  def text_alignment_df(self):
    data = []
    columns = [
      f'{self.corpus_1.name} index',
      self.corpus_1.name, 
      f'{self.corpus_2.name} index', 
      self.corpus_2.name, 
      'score'
      ]

    for match in self.best_matches:
      text_1 = self.corpus_1.corpus[match.original_index]
      text_2 = self.corpus_2.corpus[match.match_index]
      data.append([match.original_index, text_1, match.match_index, text_2, match.score])

    return pd.DataFrame(data, columns=columns)
  
  def get_matches_higher_than(self, threshold: int) -> SimilarityMatchList:
    res = SimilarityMatchList()
    for match in self.best_matches:
      if match.score > threshold:
        res.append(match)
    return res
  
  def plot_max_similarity_per_paragraph(self):
    fig, ax = plt.subplots(figsize=(35, 5))

    ax.plot([similarity.score for similarity in self.best_matches], label=f'{self.corpus_1.name} -> {self.corpus_2.name}')

    ax.set_ylim([0,1])
    ax.set_xlim([-5,325])
    ax.legend()
    plt.title(f'Max cross similarity per p: {self.corpus_1.name} -> {self.corpus_2.name}')
    plt.show()
  
  def get_best_match_of(self, index) -> SimilarityMatch:
    return self.best_matches[index]
  
  def get_bidirectional_matches_by_threshold(self, threshold, crossVersionSimilarity) -> SimilarityMatchList:
    result = SimilarityMatchList()

    for match in self.best_matches:
      match_from_another_side = crossVersionSimilarity.get_best_match_of(match.match_index)

      if match.score < threshold: continue
      if match_from_another_side.score < threshold: continue
      if match_from_another_side.match_index != match.original_index: continue

      result.append(match)

    return result

class CrossVersionSimilarity5Gram(CrossVersionSimilarity):
  def __init__(self, corpus_1, corpus_2):
    super().__init__(corpus_1, corpus_2, thesisTfIdfNgramFeatures.TfIdf5GramCharFeatures())

class CrossVersionSimilarity8Gram(CrossVersionSimilarity):
  def __init__(self, corpus_1, corpus_2):
    super().__init__(corpus_1, corpus_2, thesisTfIdfNgramFeatures.create_8_gram)

