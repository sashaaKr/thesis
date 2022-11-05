import numpy as np
import pandas as pd
from similarities.cosine import CrossVersionSimilarity

class CrossVersionSimilarityStats:
  def __init__(self, similarity: CrossVersionSimilarity):
    self.similarity = similarity
    self.matrix = np.matrix(similarity.raw_matches)
    self.calculate()
    
  def calculate(self):
    only_similarity = []
    for k in self.similarity.all_matches_without_self:
      only_similarity.append([i[1] for i in k])

    matrix = np.matrix(only_similarity)
    max_matrix = np.matrix(matrix.max(1))
    min_matrix =  np.matrix(matrix.min(1))
    
    self.max = matrix.max()
    self.min = matrix.min()

    self.max_mean = max_matrix.mean()
    self.min_mean = min_matrix.mean()
  
  def get(self):
    input_vals = [
      ('max', self.max),
      ('min', self.min),
      ('max mean', self.max_mean),
      ('min mean', self.min_mean)
    ]

    data = [i[1] for i in input_vals]
    columns = [i[0] for i in input_vals]
    return pd.DataFrame([data], columns=columns)
  
  def get_cross(self):
    input_vals = [
      ('max', self.cross_max),
      ('min', self.cross_min),
      ('max mean', self.cross_max_mean),
      ('min mean', self.cross_min_mean)
      ]

    data = [i[1] for i in input_vals]
    columns = [i[0] for i in input_vals]
    return pd.DataFrame([data], columns=columns)

  def calculate_bi_directional(self, similarity_2):
    only_similarity = []
    for k in self.similarity.get_bidirectional_strongly_similar(similarity_2):
      only_similarity.append(k.score)
    
    matrix = np.matrix(only_similarity)
    max_matrix = np.matrix(matrix.max(1))
    min_matrix =  np.matrix(matrix.min(1))

    self.cross_max = matrix.max()
    self.cross_min = matrix.min()

    self.cross_max_mean = max_matrix.mean()
    self.cross_min_mean = min_matrix.mean() 