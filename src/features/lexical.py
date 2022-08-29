import pandas as pd
import text_cleanup.text_cleanup as thesisCleanUp

def get_lexical_features(corpus):
  text = ' '.join(corpus)
  tokens = text.split()
    
  total_characters = len(text)
  total_words = len(tokens)
  unique_words = len(set(tokens))
  unique_lemmatized_word = len(thesisCleanUp.create_lemmatized_tokens(tokens))
  paragraphs = len(corpus)
    
  return total_characters, total_words, unique_words, paragraphs, unique_lemmatized_word

def create_lexical_features_df(text, label):
  df = pd.DataFrame(dtype=float)
  total_characters, total_words, unique_words, paragraphs, unique_lemmatized_word = get_lexical_features(text)
    
  df.loc[label, 'total_characters'] = total_characters
  df.loc[label, 'total_words'] = total_words
  df.loc[label, 'unique_words'] = unique_words
  df.loc[label, 'paragraphs'] = paragraphs
  df.loc[label, 'unique_lemmatized_word'] = unique_lemmatized_word
    
  return df
