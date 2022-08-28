import enum
import os
import re

import numpy as np
import pandas as pd

import utils.utils as thesisUtils
import text_cleanup.text_cleanup as thesisCleanUp
import vocabulary.vocabulary as thesisVocabulary
from sklearn.feature_extraction.text import CountVectorizer
import similarities.cosine as thesisCosineSimilarity
from collections import Counter

ZWICKAU_FILE_NAME = "A_Zwickau_RB_I_XII_5 (12).txt"
A_ZWICKAU_WITH_SECTION_SEPARATION_FILE_NAME = "A_Zwickau_RB_I_XII_5 with section separation.txt"

LONDON_FILE_NAME = "B_London_BL_Add_18929 (6).txt"
B_LONDON_WITH_SECTION_SEPARATION_FILE_NAME = "B_London_BL_Add_18929 with section separation.txt"

BRESLAU_FILE_NAME = 'SV_Breslau_221-transcription2.txt'

STOP_WORDS_FILE_NAME = "stop_words.json"

SECTION_SEPARATOR = '*********** SECTION ***********'

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.join(USER_HOME_DIR, 'thesis') 

LONG_P_THRESHOLD = 20

def get_data_file_path(file_name):
    return os.path.join(ROOT, 'full', file_name)

a_zwickau_file_path = get_data_file_path(ZWICKAU_FILE_NAME)
b_london_file_path = get_data_file_path(LONDON_FILE_NAME)
breslau_file_path = get_data_file_path(BRESLAU_FILE_NAME)

stop_words_file_path = get_data_file_path(STOP_WORDS_FILE_NAME)

def read_file(file_path):
  return open(file_path, encoding='utf-8').read()

def read_zwickau():
  return read_file(a_zwickau_file_path)

def read_zwickau_poorly_similar_with_chops():
  file_path = os.path.join(ROOT, 'computed_data', 'corpus', 'zwickau', 'zwickau_poorly_similar_with_chops.txt')
  return read_file(file_path)

def read_london_poorly_similar_with_chops():
  file_path = os.path.join(ROOT, 'computed_data', 'corpus', 'london', 'london_poorly_similar_with_chops.txt')
  return read_file(file_path)

def get_zwickau_poorly_similar_with_chops_corpus():
  raw_text = read_zwickau_poorly_similar_with_chops()
  return thesisCleanUp.tokenize_text(raw_text)

def get_london_poorly_similar_with_chops_corpus():
  raw_text = read_london_poorly_similar_with_chops()
  return thesisCleanUp.tokenize_text(raw_text)

def get_london_poorly_similar_with_chops_long_p_corpus():
  return list(filter(lambda x: len(x.split()) > 20, get_london_poorly_similar_with_chops_corpus()))

def get_zwickau_poorly_similar_with_chops_long_p_corpus():
  return list(filter(lambda x: len(x.split()) > 20, get_zwickau_poorly_similar_with_chops_corpus()))

def get_burchard_candidate_version_based_on_strongly_similar_london_base_long_p_corpus():
  return list(filter(lambda x: len(x.split()) > 20, get_burchard_candidate_version_based_on_strongly_similar_london_base()))

def get_london_poorly_similar_with_chops_with_placeholder_for_empty_sentences():
  raw_text = read_london_poorly_similar_with_chops()
  result = []

  for p in raw_text.split('\n'):
    result.append('___EMPTY___PARAGRAPH___PLACEHOLDER' if p == '' else p)

  return thesisCleanUp.tokenize_text('\n'.join(result))

# def get_zwickau_poorly_similar_with_chops_with_placeholder_for_empty_sentences():
#   raw_text = read_zwickau_poorly_similar_with_chops()
#   corpus_with_placeholders = put_empty_placeholder(raw_text)
#   return thesisCleanUp.tokenize_text('\n'.join(corpus_with_placeholders))

#   # for p in raw_text.split('\n'):
#   #   result.append('___EMPTY___PARAGRAPH___PLACEHOLDER' if p == '' else p)

#   # return thesisCleanUp.tokenize_text('\n'.join(result))

# def put_empty_placeholder(raw_text):
#   result = []
#   for p in raw_text.split('\n'):
#     result.append('___EMPTY___PARAGRAPH___PLACEHOLDER' if p == '' else p)
#   return result 

def read_zwickau_with_section_separation():
  return read_file(get_data_file_path(A_ZWICKAU_WITH_SECTION_SEPARATION_FILE_NAME))

def read_london():
  return read_file(b_london_file_path)

def read_breslau():
  return read_file(breslau_file_path)

def read_london_with_section_separation():
  return read_file(get_data_file_path(B_LONDON_WITH_SECTION_SEPARATION_FILE_NAME))

# TODO: remove this function
def get_london_corpus():
  london_text = read_london()
  london_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(london_text))
  return london_corpus

# TODO: remove this function
def get_london_by_new_line():
  return get_london_corpus()

def get_london_by_new_line_without_words_processing():
  london_text = read_london()
  london_corpus = thesisCleanUp.create_corpus_by_line_without_word_replacements(london_text)
  return london_corpus

def get_zwickau_by_new_line_without_words_processing():
  zwickau_text = read_zwickau()
  zwickau_corpus = thesisCleanUp.create_corpus_by_line_without_word_replacements(zwickau_text)
  return zwickau_corpus

# TODO: remove this function
def get_zwickau_corpus():
  zwickau_text = read_zwickau()
  zwickau_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(zwickau_text))
  return zwickau_corpus

# TODO: remove this function
def get_zwickau_by_new_line():
  return get_zwickau_corpus()

def get_breslau_corpus():
  breslau_text = read_breslau()
  breslau_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(breslau_text))
  return breslau_corpus

def get_breslau_by_new_line():
  return get_breslau_corpus()


###  CORPUSES BY 3 SENTENCES  ###
def get_zwickau_by_3_sentences():
    zwickau_text = read_zwickau()
    zwickau_corpus = thesisCleanUp.create_corpus_by_3_sentences(thesisCleanUp.jvtext(zwickau_text))
    return zwickau_corpus

def get_london_by_3_sentences():
    london_text = read_london()
    london_corpus = thesisCleanUp.create_corpus_by_3_sentences(thesisCleanUp.jvtext(london_text))
    return london_corpus

def get_breslau_by_3_sentences():
    breslau_text = read_breslau()
    breslau_corpus = thesisCleanUp.create_corpus_by_3_sentences(thesisCleanUp.jvtext(breslau_text))
    return breslau_corpus


###  CORPUSES BY 2 SENTENCES  ###
def get_zwickau_by_2_sentences():
    zwickau_text = read_zwickau()
    zwickau_corpus = thesisCleanUp.create_corpus_by_2_sentences(thesisCleanUp.jvtext(zwickau_text))
    return zwickau_corpus

def get_london_by_2_sentences():
    london_text = read_london()
    london_corpus = thesisCleanUp.create_corpus_by_2_sentences(thesisCleanUp.jvtext(london_text))
    return london_corpus

def get_breslau_by_2_sentences():
    breslau_text = read_breslau()
    breslau_corpus = thesisCleanUp.create_corpus_by_2_sentences(thesisCleanUp.jvtext(breslau_text))
    return breslau_corpus




def get_zwickau_separated_by_sections():
    zwickau_text_with_separation = read_zwickau_with_section_separation()
    zwickau_sections = zwickau_text_with_separation.split(SECTION_SEPARATOR)
    return zwickau_sections

def get_london_separated_by_sections():
    london_text_with_separation = read_london_with_section_separation()
    london_sections = london_text_with_separation.split(SECTION_SEPARATOR)
    return london_sections

def get_zwickau_section_indexes():
    zwickau_sections_indexes = np.zeros(322)
    zwickau_sections_indexes[0:20] = 1
    zwickau_sections_indexes[20:20+249] = 2
    zwickau_sections_indexes[20+249:20+249+7] = 3
    zwickau_sections_indexes[20+249+7:20+249+7+21] = 4
    zwickau_sections_indexes[20+249+7+21:20+249+7+21+25] = 5
    return zwickau_sections_indexes

def get_london_section_indexes():
    london_sections_indexes = np.zeros(318)
    london_sections_indexes[0:20] = 1
    london_sections_indexes[20:20+242] = 2
    london_sections_indexes[20+242:20+242+6] = 3
    london_sections_indexes[20+242+6:20+242+6+21] = 4
    london_sections_indexes[20+242+6+21:20+242+6+21+29] = 5
    return london_sections_indexes


# it seems that this function is redundant cause for burchard version we should use only strongly similar paragraphs
# def get_burchard_candidate_version_based_on_p_aligment_london_base():
#   london_zwickau_breslau_p_aligment_df = pd.read_csv('../computed_data/p_aligment/by_new_line/london_zwickau_breslau.csv').drop(['Unnamed: 0'], axis=1)

#   result = []

#   for index, row in london_zwickau_breslau_p_aligment_df.iterrows():
#     london_p = row['london text']
#     zwickau_p = row['zwickau text']

#     shared_words = []

#     # TODO: interesting if it is matter which version to split and run the same function but split zwickau will provide different result
#     for word in london_p.split():
#       match_in_london = re.search(r'\b' + word + r'\b', london_p)
#       match_in_zwickau = re.search(r'\b' + word + r'\b', zwickau_p)

#       if match_in_london and match_in_zwickau:
#         shared_words.append(word)

#     result.append(' '.join(shared_words))

#   return result

def get_london_zwickau_strongly_similar_by_new_line_df():
  return pd.read_csv('../computed_data/p_aligment/by_new_line/strongly_similar/london_zwickau_breslau.csv').drop(['Unnamed: 0'], axis=1)

def get_burchard_candidate_version_based_on_strongly_similar_london_base():
  london_zwickau_breslau_p_aligment_df = get_london_zwickau_strongly_similar_by_new_line_df()
  result = []

  for index, row in london_zwickau_breslau_p_aligment_df.iterrows():
    london_p = row['london text']
    zwickau_p = row['zwickau text']
    result.append(' '.join(thesisUtils.get_shared_words(london_p, zwickau_p)))

    # shared_words = []

    # # TODO: interesting if it is matter which version to split and run the same function but split zwickau will provide different result
    # for word in london_p.split():
    #   match_in_london = re.search(r'\b' + word + r'\b', london_p)
    #   match_in_zwickau = re.search(r'\b' + word + r'\b', zwickau_p)

    #   if match_in_london and match_in_zwickau:
    #     shared_words.append(word)

    # result.append(' '.join(shared_words))

  return result

def get_burchard_candidate_version_based_on_strongly_similar_zwickau_base():
  london_zwickau_breslau_p_aligment_df = pd.read_csv('../computed_data/p_aligment/by_new_line/strongly_similar/zwickau_london_breslau.csv').drop(['Unnamed: 0'], axis=1)
  result = []

  for index, row in london_zwickau_breslau_p_aligment_df.iterrows():
    london_p = row['london text']
    zwickau_p = row['zwickau text']

    shared_words = []

    # TODO: interesting if it is matter which version to split and run the same function but split zwickau will provide different result
    for word in zwickau_p.split():
      match_in_london = re.search(r'\b' + word + r'\b', london_p)
      match_in_zwickau = re.search(r'\b' + word + r'\b', zwickau_p)

      if match_in_london and match_in_zwickau:
        shared_words.append(word)

    result.append(' '.join(shared_words))

  return result

def get_burchard_candidate_version_based_on_p_aligment_zwickau_base():
  london_zwickau_breslau_p_aligment_df = pd.read_csv('../computed_data/p_aligment/by_new_line/zwickau_london_breslau.csv').drop(['Unnamed: 0'], axis=1)

  result = []

  for index, row in london_zwickau_breslau_p_aligment_df.iterrows():
    london_p = row['london text']
    zwickau_p = row['zwickau text']

    shared_words = []

    # TODO: interesting if it is matter which version to split and run the same function but split zwickau will provide different result
    for word in zwickau_p.split():
      match_in_london = re.search(r'\b' + word + r'\b', london_p)
      match_in_zwickau = re.search(r'\b' + word + r'\b', zwickau_p)

      if match_in_london and match_in_zwickau:
        shared_words.append(word)

    result.append(' '.join(shared_words))

  return result

def get_london_zwickau_breslau_strongly_similar_p_aligment_df():
  return pd.read_csv('../computed_data/p_aligment/by_new_line/strongly_similar/london_zwickau_breslau.csv').drop(['Unnamed: 0'], axis=1)

# def get_london_zwickau_chop_from_strongly_similar():
#   london_zwickau_breslau_strongly_similar_df = get_london_zwickau_breslau_strongly_similar_p_aligment_df()

#   result = []

def create_burchard_index_map_to_original_versions():
  london_zwickau_breslau_p_aligment_df = get_london_zwickau_strongly_similar_by_new_line_df()
  london_p_to_index_dict = thesisUtils.p_to_index_dictionary(get_london_by_new_line())
  zwickau_p_to_index_dict = thesisUtils.p_to_index_dictionary(get_zwickau_by_new_line())

  burchard_to_versions_dict = {}
  for index, row in london_zwickau_breslau_p_aligment_df.iterrows():
    london_p = row['london text']
    zwickau_p = row['zwickau text']
    
    london_original_index = london_p_to_index_dict[london_p]
    zwickau_original_index = zwickau_p_to_index_dict[zwickau_p]

    burchard_to_versions_dict[index] = { 'london': london_original_index, 'zwickau': zwickau_original_index  }
  
  return burchard_to_versions_dict

def create_burchard_to_versions_text_to_text_map():
  london_corpus = get_london_by_new_line_without_words_processing()
  zwickau_corpus = get_zwickau_by_new_line_without_words_processing()
  burchard_candidate = thesisVocabulary.create_pre_proceed_corpus_from_processed_corpus(
    get_burchard_candidate_version_based_on_strongly_similar_london_base(),
    thesisVocabulary.create_london_pre_post_processing_map()
  )
  burchard_to_versions_index_map = create_burchard_index_map_to_original_versions()

  result = []
  for burchard_index, versions_indexes in burchard_to_versions_index_map.items():
    london_index = versions_indexes['london']
    zwickau_index = versions_indexes['zwickau']

    burchard_text = burchard_candidate[burchard_index]
    london_text = london_corpus[london_index]
    zwickau_text = zwickau_corpus[zwickau_index]

    result.append([burchard_text, london_text, zwickau_text]) 

  return result

def get_burchard_candidate_version_based_on_strongly_similar_london_base_long_p():
  burchard_candidate_version_based_london_without_word_processing = thesisVocabulary.create_pre_proceed_corpus_from_processed_corpus(
    get_burchard_candidate_version_based_on_strongly_similar_london_base(),
    thesisVocabulary.create_london_pre_post_processing_map()
  )

  print(f'Original corpus length: {len(burchard_candidate_version_based_london_without_word_processing)}')
  burchard_candidate_version_based_london_without_word_processing_long_p = list(filter(lambda x: len(x.split()) > LONG_P_THRESHOLD, burchard_candidate_version_based_london_without_word_processing))
  print(f'Long p corpus length: {len(burchard_candidate_version_based_london_without_word_processing_long_p)}')

  return burchard_candidate_version_based_london_without_word_processing_long_p

def get_london_poorly_similar_with_chops_corpus_without_word_processing_long_p():
  london_poorly_similar_with_chops_corpus_without_word_processing = thesisVocabulary.create_pre_proceed_corpus_from_processed_corpus(
    get_london_poorly_similar_with_chops_corpus(),
    thesisVocabulary.create_london_pre_post_processing_map()
  )

  print(f'Original corpus length: {len(london_poorly_similar_with_chops_corpus_without_word_processing)}')
  london_poorly_similar_with_chops_corpus_without_word_processing_long_p = list(filter(lambda x: len(x.split()) > LONG_P_THRESHOLD, london_poorly_similar_with_chops_corpus_without_word_processing))
  print(f'Long p corpus length: {len(london_poorly_similar_with_chops_corpus_without_word_processing_long_p)}')

  return london_poorly_similar_with_chops_corpus_without_word_processing_long_p

def get_zwickau_poorly_similar_with_chops_corpus_without_word_processing_long_p():
  zwickau_poorly_similar_with_chops_corpus_without_word_processing = thesisVocabulary.create_pre_proceed_corpus_from_processed_corpus(
    get_zwickau_poorly_similar_with_chops_corpus(),
    thesisVocabulary.create_zwickau_pre_post_processing_map()
  )

  print(f'Original corpus length: {len(zwickau_poorly_similar_with_chops_corpus_without_word_processing)}')
  zwickau_poorly_similar_with_chops_corpus_without_word_processing_long_p = list(filter(lambda x: len(x.split()) > LONG_P_THRESHOLD, zwickau_poorly_similar_with_chops_corpus_without_word_processing))
  print(f'Long p corpus length: {len(zwickau_poorly_similar_with_chops_corpus_without_word_processing_long_p)}')

  return zwickau_poorly_similar_with_chops_corpus_without_word_processing_long_p 

def filter_short_p(corpus):
    return list(filter(lambda x: len(x.split()) > 20, corpus))

class Corpus:
  def __init__(self, path, name):
    self.path = path
    self.name = name

    self.raw_text = self.read()
    self.corpus = self.text_processing()

  def read(self): 
    return open(self.path, encoding='utf-8').read()
  
  def get_n_grams_words_dictionary(self, ngram_from=2, ngram_to=2):    
    vec = CountVectorizer(
        ngram_range = (ngram_from, ngram_to),
        token_pattern = r"(?u)\b\w+\b"
    ).fit(self.corpus)
    bag_of_words = vec.transform(self.corpus)
    sum_words = bag_of_words.sum(axis = 0) 
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

    words_freq_dic = {}
    for i in words_freq:
        [word, count] = i
        words_freq_dic[word] = count
    return words_freq_dic

  # here is neat graph for count: https://www.absentdata.com/python-graphs/python-word-frequency/
  def words_frequency(self):    
    word_counter = Counter(self.raw_text.split())
    return sorted(word_counter.items(), key = lambda item: item[1], reverse = True) 

  def get_shared_n_grams(self, corpus, ngram):
    self_n_gram_dictionary = self.get_n_grams_words_dictionary(ngram, ngram)
    another_corpus_n_gram_dictionary = corpus.get_n_grams_words_dictionary(ngram, ngram)

    shared_words = {}
    
    for (word, counter) in self_n_gram_dictionary.items():
      if another_corpus_n_gram_dictionary.get(word) is not None:
        result = {}
        result[self.name] = counter
        result[corpus.name] = another_corpus_n_gram_dictionary[word]
        shared_words[word] = result 
    
    return shared_words

class CorpusByNewLine(Corpus):
  @staticmethod
  def london():
    return CorpusByNewLine(get_data_file_path(LONDON_FILE_NAME), 'london')

  @staticmethod
  def zwickau():
    return CorpusByNewLine(get_data_file_path(ZWICKAU_FILE_NAME), 'zwickau')
  
  @staticmethod
  def breslau():
    return CorpusByNewLine(get_data_file_path(BRESLAU_FILE_NAME), 'breslau')
  
  def __init__(self, path, name):
    super().__init__(path, name)

  def text_processing(self):
    return thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(self.raw_text))

class BurchardCorpus:
  def __init__(self, corpus_1, corpus_2):
    self.name = f'burchard_candidate_by_{corpus_1.name}_{corpus_2.name}'
    self.corpus_1 = corpus_1
    self.corpus_2 = corpus_2
    self.similarities_1 = thesisCosineSimilarity.CrossVersionSimilarity5Gram(corpus_1, corpus_2)
    self.similarities_2 = thesisCosineSimilarity.CrossVersionSimilarity5Gram(corpus_2, corpus_1)
    self.similarities_1.load()
    self.similarities_2.load()

    corpus, matches_used_for_build = self.build_corpus()
    self.corpus = corpus
    self.matches_used_for_build = matches_used_for_build
  
  def build_corpus(self):
    corpus = []
    matches_used_for_build = []
    for match in self.similarities_1.get_bidirectional_strongly_similar(self.similarities_2):
      corpus.append(' '.join(thesisUtils.get_shared_words(match.original_text, match.match_text)))
      matches_used_for_build.append(match)
    return corpus, matches_used_for_build
  
  def corpus_for_predictions(self):
    return filter_short_p(self.corpus)

  def with_classifier_predictions(self, wrong_predictions_1, wrong_predictions_2):
    is_burchard = True
    temp_corpus = [ [p, is_burchard, is_burchard] for p in self.corpus_for_predictions() ]

    for prediction in wrong_predictions_1:
      temp_corpus[prediction.index][1] = False
    for prediction in wrong_predictions_2:
      temp_corpus[prediction.index][2] = False

    return temp_corpus
  
  def get_burchard_predicted_truly(self, wrong_predictions_1, wrong_predictions_2):
    with_classifiers_result = self.with_classifier_predictions(wrong_predictions_1, wrong_predictions_2)
    result = []
    for index, item in enumerate(with_classifiers_result):
      if item[1] and item[2]:
        result.append({ 'text': item[0], 'index': index  })
    
    return result

class LeftoversCorpus:
  def __init__(self, corpus_1, corpus_2):
    self.name = f'leftovers_{corpus_1.name}'
    self.corpus_1 = corpus_1
    self.corpus_2 = corpus_2
    self.similarities_1 = thesisCosineSimilarity.CrossVersionSimilarity5Gram(corpus_1, corpus_2)
    self.similarities_2 = thesisCosineSimilarity.CrossVersionSimilarity5Gram(corpus_2, corpus_1)
    self.similarities_1.load()
    self.similarities_2.load()
    self.corpus = self.build_corpus()

  def corpus_for_predictions(self):
    return filter_short_p(self.corpus)

  def corpus_with_placeholders_for_empty(self):
    result = []
    for p in self.corpus:
      result.append('___EMPTY___PARAGRAPH___PLACEHOLDER' if p == '' else p)
    return result

  def with_classifier_predictions(self, wrong_predictions_1, wrong_predictions_2):
    temp_corpus = [ [p, True, True] for p in self.corpus_for_predictions() ]

    for prediction in wrong_predictions_1:
      temp_corpus[prediction.index][1] = False
    for prediction in wrong_predictions_2:
      temp_corpus[prediction.index][2] = False

    return temp_corpus

  def leftovers_predicted_falsy(self, wrong_predictions_1, wrong_predictions_2):
    with_classifiers_result = self.with_classifier_predictions(wrong_predictions_1, wrong_predictions_2)
    result = []
    for index, item in enumerate(with_classifiers_result):
      if not item[1] and not item[2]:
        result.append({ 'text': item[0], 'index': index  })
    
    return result

  def build_corpus(self):
    corpus = []
    strongly_similar = set(self.similarities_1.get_bidirectional_matches_by_threshold(0.5, self.similarities_2).original_indexes())

    for index, row in self.similarities_1.text_alignment_df().iterrows():
      index = row[f'{self.corpus_1.name} index']
      corpus_1_p = row[self.corpus_1.name]
      corpus_2_p = row[self.corpus_2.name]

      corpus_1_p_without_shared_words = corpus_1_p

      if index in strongly_similar:
        for word in corpus_1_p.split():
          match_in_corpus_1 = re.search(r'\b' + word + r'\b', corpus_1_p)
          match_in_corpus_2 = re.search(r'\b' + word + r'\b', corpus_2_p)
          if match_in_corpus_1 and match_in_corpus_2:
            corpus_1_p_without_shared_words = re.sub(r'\b' + word + r'\b', '', corpus_1_p_without_shared_words, count = 1).replace('  ', ' ').strip() 

      corpus.append(corpus_1_p_without_shared_words)

    return corpus
