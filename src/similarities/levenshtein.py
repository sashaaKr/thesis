from cgitb import reset
import sys
import math

from urllib3 import Retry
from leven import levenshtein
import numpy as np

def get_edit_distance(word_1, word_2):
  lev_dist = levenshtein(word_1, word_2)
  return lev_dist

def get_similarity_tupels(words_1, words_2):
    """
    Get 2 arrays of words and return a list of tuples with word from word_1, most similar word from words_2 and score
    """

    result = []
    minimal_score = sys.maxsize

    for w in words_1:
      most_similar = ''

      for j in words_2:
        lev_dist = levenshtein(w, j)
        if lev_dist < minimal_score:
          minimal_score = lev_dist
          most_similar = j
      
      result.append((w, most_similar, minimal_score))
      minimal_score = sys.maxsize
    
    result.sort(key=lambda x: x[2])
    return result


def get_sentence_score(sentence_1, sentence_2):
    """
    Get 2 sentences and return a score
    """

    words_1 = sentence_1.split()
    words_2 = sentence_2.split()

    tupels = get_similarity_tupels(words_1, words_2)

    score = 0
    for t in tupels:
      score += t[2]

    return score

def get_2_corpus_best_similarities(corpus_1, corpus_2):
  # result = np.zeros(len(corpus_1))
  result = []

  for index_corpus_1, p_corpus_1 in enumerate(corpus_1):
    most_similar_score = sys.maxsize
    best_match_index = -1

    for index_corpus_2, p_corpus_2 in enumerate(corpus_2):
      score = get_sentence_score(p_corpus_1, p_corpus_2)

      if score < most_similar_score:
        most_similar_score = score
        best_match_index = index_corpus_2
        # result[index_corpus_1] = (index_corpus_2, score)
  
    result.append((best_match_index, most_similar_score))
  return result

def get_inner_version_best_similarities(corpus):
  result = []
  for index, p in enumerate(corpus):
    corpus_without_self = [ p for i, p in enumerate(corpus) if i != index ]
    result.append(get_2_corpus_best_similarities([p], corpus_without_self))
  
  return result

def get_2_corpus_all_similarities(corpus_1, corpus_2):
  result = []

  for index_corpus_1, p_corpus_1 in enumerate(corpus_1):
    p_result = []

    for index_corpus_2, p_corpus_2 in enumerate(corpus_2):
      p_result.append(get_sentence_score(p_corpus_1, p_corpus_2))
    
    result.append(p_result)
  
  return result

def find_possible_mistakes(w, corpus):
  result = {} 
  corpus_words = set(''.join(corpus).split())

  for word in corpus_words:
    res = levenshtein(w, word)
    if res == 0: continue
    if res == len(w): continue
    if res > len(w) / 3: continue

    # if res < 5:
    if res < 4:
      result[word] = res
  
  return result

def create_version_possible_errors_mapping(corpus_1, corpus_2, error_threshold):
  words_1 = set(' '.join(corpus_1).split())
  words_2 = set(' '.join(corpus_2).split())

  result = {}

  for w_1 in words_1:
    result[w_1]  =[]

    for w_2 in words_2:
      lev_distance = levenshtein(w_1, w_2)

      if lev_distance == 0: continue
      if lev_distance >= math.trunc(len(w_1) / 2): continue

      if lev_distance <= error_threshold:
        result[w_1].append((w_2, lev_distance))

  return result

def get_p_score(corpus, corpus_error_mapping):
  result = []

  for index, p in enumerate(corpus):
    score = 0
    for word in p.split():
      if word in corpus_error_mapping and len(corpus_error_mapping[word]) > 0:
        for possible_error in corpus_error_mapping[word]:
          # print(possible_error)
          score += possible_error[1]
    
    result.append(score)
  
  return result

def get_non_empty_alternatives(alternatives):
  result = {}
  for i in alternatives:
    if len(alternatives[i]) > 0:
      result[i] = alternatives[i]
  
  return result