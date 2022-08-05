import utils.utils as thesisUtils
import data.reader as thesisDataReader
import text_cleanup.text_cleanup as thesisTextCleanUp
import text_cleanup.text_cleanup as thesisCleanUp
import pandas as pd

from collections import Counter
# # here is neat graph for count: https://www.absentdata.com/python-graphs/python-word-frequency/
def create_words_frequency(corpus):    
    word_counter =  Counter(' '.join(corpus).split())
    return sorted(word_counter.items(), key=lambda item: item[1], reverse=True)

def create_words_dictionary(corpus):
    words_frequency = create_words_frequency(corpus)
    return thesisUtils.convert_array_of_tuples_to_dics(words_frequency)


# getting main version and 2 another versions for comparisons
# return list of tuples: (word, frequency)
# for words that appears only in main version
def get_version_unique_words(main_version_dictionary, version_2_dictionary, version_3_dictionary):
    unique_word = []
    for word in main_version_dictionary:
        if main_version_dictionary[word] > 1 and version_2_dictionary.get(word) is None and version_3_dictionary.get(word) is None:
            unique_word.append(word)
    return unique_word

def get_version_shared_words(version_a_dictionary, version_a_name, version_b_dictionary, version_b_name, version_c_dictionary):
    shared_words = {}
    for word in version_a_dictionary:
        if version_b_dictionary.get(word) is not None and version_c_dictionary.get(word) is None:
            shared_words[word] = {
                version_a_name: version_a_dictionary[word],
                version_b_name: version_b_dictionary[word]
            }
    return shared_words

# Create vocabulary that shared in version a and b, but not in c - for all permutations
def get_shared_vocabulary_for_2_versions(
    corpus_a, 
    corpus_a_name,
    corpus_b, 
    corpus_b_name,
    corpus_c,
    ):
    dictionary_a = create_words_dictionary(corpus_a)
    dictionary_b = create_words_dictionary(corpus_b)
    dictionary_c = create_words_dictionary(corpus_c)

    return get_version_shared_words(
        dictionary_a,
        corpus_a_name,
        dictionary_b,
        corpus_b_name,
        dictionary_c,
    )

def unique_vocabulary():
    london_corpus = thesisDataReader.get_london_corpus()
    zwickau_corpus = thesisDataReader.get_zwickau_corpus()
    breslau_corpus = thesisDataReader.get_breslau_corpus()

    london_dictionary = create_words_dictionary(london_corpus)
    zwickau_dictionary = create_words_dictionary(zwickau_corpus)
    breslau_dictionary = create_words_dictionary(breslau_corpus)    
    
    london_unique_words = get_version_unique_words(london_dictionary, zwickau_dictionary, breslau_dictionary)
    zwickau_unique_words = get_version_unique_words(zwickau_dictionary, london_dictionary, breslau_dictionary)
    breslau_unique_words = get_version_unique_words(breslau_dictionary, london_dictionary, zwickau_dictionary)
    
    df_columns = ['version', 'word', 'count']
    df_data = []
    
    for word in london_unique_words:
        df_data.append(['london', word, london_dictionary.get(word)])
    
    for word in zwickau_unique_words:
        df_data.append(['zwickau', word, zwickau_dictionary.get(word)])
        
    for word in breslau_unique_words:
            df_data.append(['breslau', word, breslau_dictionary.get(word)])
    
    return pd.DataFrame(df_data, columns=df_columns)

def get_3_versions_shared_words(version_a_dictionary, version_a_name, version_b_dictionary, version_b_name, version_c_dictionary, version_c_name):
    shared_words = {}
    for word in version_a_dictionary:
        if version_b_dictionary.get(word) is not None and version_c_dictionary.get(word) is not None:
            shared_words[word] = {
                version_a_name: version_a_dictionary[word],
                version_b_name: version_b_dictionary[word],
                version_c_name: version_c_dictionary[word],
            }
    return shared_words

def create_pre_post_processing_map(raw_text):    
  data = []
  counts = Counter(''.join(raw_text.split('\n')).split(' '))

  for word in counts:
    resp = thesisTextCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(word))
    if len(resp) == 0: continue
    data.append([word, resp[0]])

  return pd.DataFrame(data, columns=['before', 'after'])

def create_london_pre_post_processing_map():
  return create_pre_post_processing_map(' \n'.join(thesisDataReader.get_london_by_new_line_without_words_processing()))

def create_zwickau_pre_post_processing_map():
  return create_pre_post_processing_map(' \n'.join(thesisDataReader.get_zwickau_by_new_line_without_words_processing()))

def create_pre_proceed_corpus_from_processed_corpus(processed_corpus, pre_post_processing_map):
  result = []

  for p in processed_corpus:
    new_p = ''

    for word in p.split():
      original_word = pre_post_processing_map[pre_post_processing_map['after'] == word]['before'].values[0]
      new_p = f'{new_p} {original_word}' if len(new_p) > 0 else original_word

    result.append(new_p)
  
  return result

def create_post_pre_map_for_london_poorly_similar_with_chops_with_placeholder_for_empty_sentences():
  original_london_corpus = thesisDataReader.get_london_by_new_line_without_words_processing() 
  poorly_with_chops_corpus = thesisDataReader.get_london_poorly_similar_with_chops_with_placeholder_for_empty_sentences()
  return zip(poorly_with_chops_corpus, original_london_corpus) 

def create_post_pre_map_for_zwickau_poorly_similar_with_chops_with_placeholder_for_empty_sentences():
  original_zwickau_corpus = thesisDataReader.get_zwickau_by_new_line_without_words_processing() 
  # in zwickau no empty sentences in poorly similar with chops corpus
  poorly_with_chops_corpus = thesisDataReader.get_zwickau_poorly_similar_with_chops_corpus()
  return zip(poorly_with_chops_corpus, original_zwickau_corpus) 
