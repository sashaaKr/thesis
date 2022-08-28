import numpy as np
import re

def get_n_indexes_of_max_values(arr, n):
    indices = np.argpartition(-arr, n)[:n]
    sorted_indices = indices[np.argsort(arr[indices])]
    return sorted_indices

def get_max_similarity_per_p(similarities):
    res = []
    for index, value in enumerate(similarities):
        max_indices = get_n_indexes_of_max_values(value, 6)
        max_indices_without_self = max_indices[:-1]
        max_similarity = value[max_indices_without_self[-1]]
        res.append(max_similarity)
    return res

def convert_array_of_tuples_to_dics(arr):
    dict = {}
    for a, b in arr:
        dict.setdefault(a, b)
    return dict

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if i + (2*n) > len(lst):
            yield lst[i:len(lst)]
            break
        else: yield lst[i:i + n]

def find_text_p_indexed_in_corpus(corpus, text_to_find):
  found_indexes = []
  for index, paragraph in enumerate(corpus):
    match = re.search(r'\b' + text_to_find + r'\b', paragraph)
    if match: found_indexes.append(index)
    
  return found_indexes

def p_to_index_dictionary(corpus):
  return { p: i for i, p in enumerate(corpus) }

def get_shared_words(text1, text2):
  shared_words = []
  for word in text1.split():
    match_in_text1 = re.search(r'\b' + word + r'\b', text1)
    match_in_text2 = re.search(r'\b' + word + r'\b', text2)
    if match_in_text1 and match_in_text2:
      shared_words.append(word)
  
  return shared_words

def flatten(arr):
  return [item for sublist in arr for item in sublist] 