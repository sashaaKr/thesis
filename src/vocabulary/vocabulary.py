import utils.utils as thesisUtils
import data.reader as thesisDataReader
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

# Create vocabulary that shared in vesion a and b, but not in c - for all permutations
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