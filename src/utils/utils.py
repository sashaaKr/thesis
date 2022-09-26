import numpy as np
import re
import difflib

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

# TODO: move shared words indexes to dedicated function
def get_shared_words(text1, text2, include_indexes = False):
  _text1 = text1
  _text2 = text2
  shared_words = []
  removes = 0

  for word in text1.split():
    match_in_text1 = find_word(word, _text1)
    match_in_text2 = find_word(word, _text2)

    if match_in_text1 and match_in_text2:
      if include_indexes:
        word_len = len(word)
        print(match_in_text2.start())
        shared_words.append(
          (
            word,
            (match_in_text1.start() + removes, match_in_text1.start() + word_len + removes),
            (match_in_text2.start() + removes, match_in_text2.start() + word_len + removes)
          )
        ) 
        removes += word_len + 1 # whitespace
      else: shared_words.append(word)

      _text1 = remove_one_word(word, _text1)
      _text2 = remove_one_word(word, _text2)
  
  return shared_words

def get_indexes_of_shard_word(text1, text2):
  result = []
  was_found_in_text1 = {}
  was_found_in_text2 = {}

  _text1 = text1
  _text2 = text2

  for word in text1.split():
    match_in_text1 = find_all_word_appearances(word, text1)
    match_in_text2 = find_all_word_appearances(word, text2)

    if find_word(word, _text1) and find_word(word, _text2):
      _text1 = remove_one_word(word, _text1)
      _text2 = remove_one_word(word, _text2)
      
      if word not in was_found_in_text1: was_found_in_text1[word] = 0
      else: was_found_in_text1[word] += 1

      if word not in was_found_in_text2: was_found_in_text2[word] = 0
      else: was_found_in_text2[word] += 1

      found1 = was_found_in_text1[word]
      found2 = was_found_in_text2[word]

      r1 = match_in_text1[found1]
      r2 = match_in_text2[found2]
      
      result.append((
        word,
        (r1.start(), r1.end()),
        (r2.start(), r2.end())
      ))

  return result

def find_word(word, str):
  return re.search(r'\b' + word + r'\b', str)

def find_all_word_appearances(word, str):
  return list(re.finditer(r'\b' + word + r'\b', str))

def remove_one_word(word_to_remove, text):
  return (
    re
    .sub(r'\b' + word_to_remove + r'\b', '', text, count = 1)
    .replace('  ', ' ')
    .strip()
    )

def flatten(arr):
  return [item for sublist in arr for item in sublist] 