import numpy as np
import pandas as pd

from data.reader import Corpus

class CorpusStats:
  def __init__(self, corpus: Corpus):
    self.corpus = corpus
    self.calculate()

  def calculate(self): 
    self.paragraphs = len(self.corpus.corpus)
    self.total_words = len(' '.join(self.corpus.corpus).split())
    self.unique_words = len(set(' '.join(self.corpus.corpus).split()))
    self.avg_paragraph_word_len = np.average([len(i.split()) for i in self.corpus.corpus])
    self.avg_paragraph_char_len = np.average([ len("".join(i.split())) for i in self.corpus.corpus])


  def get(self):
    columns = ['paragraphs', 'Total words', 'Unique words', 'Avg. paragraph length (words)', 'Avg. paragraph length (characters)']
    data = [[self.paragraphs, self.total_words, self.unique_words, self.avg_paragraph_word_len, self.avg_paragraph_char_len]]

    return pd.DataFrame(data, columns=columns, index = [[self.corpus.name]])