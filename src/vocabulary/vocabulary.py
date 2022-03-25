import utils.utils as thesisUtils

from collections import Counter
# # here is neat graph for count: https://www.absentdata.com/python-graphs/python-word-frequency/
def create_words_frequency(corpus):    
    word_counter =  Counter(' '.join(corpus).split())
    return sorted(word_counter.items(), key=lambda item: item[1], reverse=True)

def create_words_dictionary(corpus):
    words_frequency = create_words_frequency(corpus)
    return thesisUtils.convert_array_of_tuples_to_dics(words_frequency)