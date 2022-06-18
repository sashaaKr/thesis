import os
import re
import text_cleanup.text_cleanup as thesisCleanUp
import numpy as np
import pandas as pd

A_ZWICKAU_FILE_NAME = "A_Zwickau_RB_I_XII_5 (12).txt"
A_ZWICKAU_WITH_SECTION_SEPARATION_FILE_NAME = "A_Zwickau_RB_I_XII_5 with section separation.txt"

B_LONDON_FILE_NAME = "B_London_BL_Add_18929 (6).txt"
B_LONDON_WITH_SECTION_SEPARATION_FILE_NAME = "B_London_BL_Add_18929 with section separation.txt"

BREALAU_FILE_NAME = 'SV_Breslau_221-transcription2.txt'

STOP_WORDS_FILE_NAME = "stop_words.json"

SECTION_SEPARATOR = '*********** SECTION ***********'

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.join(USER_HOME_DIR, 'thesis',) 

def get_data_file_path(file_name):
    return os.path.join(ROOT, 'full', file_name)

a_zwickau_file_path = get_data_file_path(A_ZWICKAU_FILE_NAME)
b_london_file_path = get_data_file_path(B_LONDON_FILE_NAME)
breslau_file_path = get_data_file_path(BREALAU_FILE_NAME)

stop_words_file_path = get_data_file_path(STOP_WORDS_FILE_NAME)

def read_file(file_path):
    return open(file_path, encoding='utf-8').read()

def read_zwickau():
    return read_file(a_zwickau_file_path)

def read_zwickau_with_section_separation():
    return read_file(get_data_file_path(A_ZWICKAU_WITH_SECTION_SEPARATION_FILE_NAME))

def read_london():
    return read_file(b_london_file_path)

def read_breslau():
    return read_file(breslau_file_path)

def read_london_with_section_separation():
    return read_file(get_data_file_path(B_LONDON_WITH_SECTION_SEPARATION_FILE_NAME))

def get_london_corpus():
  london_text = read_london()
  london_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(london_text))
  return london_corpus

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

def get_zwickau_corpus():
  zwickau_text = read_zwickau()
  zwickau_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(zwickau_text))
  return zwickau_corpus

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

def get_burchard_candidate_version_based_on_strongly_similar_london_base():
  london_zwickau_breslau_p_aligment_df = pd.read_csv('../computed_data/p_aligment/by_new_line/strongly_similar/london_zwickau_breslau.csv').drop(['Unnamed: 0'], axis=1)
  result = []

  for index, row in london_zwickau_breslau_p_aligment_df.iterrows():
    london_p = row['london text']
    zwickau_p = row['zwickau text']

    shared_words = []

    # TODO: interesting if it is matter which version to split and run the same function but split zwickau will provide different result
    for word in london_p.split():
      match_in_london = re.search(r'\b' + word + r'\b', london_p)
      match_in_zwickau = re.search(r'\b' + word + r'\b', zwickau_p)

      if match_in_london and match_in_zwickau:
        shared_words.append(word)

    result.append(' '.join(shared_words))

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