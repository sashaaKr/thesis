from gensim import corpora

def create_bag_of_words(corpus):
  processed_corpus = [ [ word for word in document.split() ] for document in corpus] 
  dictionary = create_dictionary(corpus)
  return [dictionary.doc2bow(text) for text in processed_corpus]

def create_dictionary(corpus):
  processed_corpus = [ [ word for word in document.split() ] for document in corpus] 
  return corpora.Dictionary(processed_corpus)