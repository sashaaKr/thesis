from lib2to3.pgen2.pgen import DFAState
import data.reader as dataReader
import features.model_features as thesisModelFeatures
from data.reader import Corpus

class FeaturesFactory:
  def __init__(
    self,
    *,
    london_corpus: Corpus,
    zwickau_corpus: Corpus,
    ):
    self.london = london_corpus
    self.zwickau = zwickau_corpus

    self.london_leftovers = dataReader.LeftoversCorpus(london_corpus, zwickau_corpus)
    self.zwickau_leftovers = dataReader.LeftoversCorpus(zwickau_corpus, london_corpus)

    self.burchard_corpus_by_london = dataReader.BurchardCorpus(london_corpus, zwickau_corpus)
    self.burchard_corpus_by_zwickau = dataReader.BurchardCorpus(zwickau_corpus, london_corpus)

  def burchard_by_zwickau_VS_zwickau(self):
    df, vectorizer = self.__burchard_VS_zwickau(self.burchard_corpus_by_zwickau.filter_short_p())
    self.burchard_by_zwickau_VS_zwickau_features_df = df 
    self.burchard_by_zwickau_VS_zwickau_vectorizer = vectorizer
    return self.burchard_by_zwickau_VS_zwickau_features_df

  def burchard_by_london_VS_zwickau(self):
    df, vectorizer = self.__burchard_VS_zwickau(self.burchard_corpus_by_london.filter_short_p())
    self.burchard_by_london_VS_zwickau_features_df = df
    self.burchard_by_london_VS_zwickau_vectorizer = vectorizer
    return self.burchard_by_london_VS_zwickau_features_df

  def burchard_by_london_VS_london(self):
    df, vectorizer = self.__burchard_VS_london(self.burchard_corpus_by_london.filter_short_p())
    self.burchard_by_london_VS_london_features_df = df
    self.burchard_by_london_VS_london_vectorizer = vectorizer
    return self.burchard_by_london_VS_london_features_df

  def burchard_by_zwickau_VS_london(self):
    df, vectorizer = self.__burchard_VS_london(self.burchard_corpus_by_zwickau.filter_short_p())
    self.burchard_by_zwickau_VS_london_features_df = df
    self.burchard_by_zwickau_VS_london_vectorizer = vectorizer
    return self.burchard_by_zwickau_VS_london_features_df
  
  def london_VS_zwickau(self, *, n_gram = (2, 5)):
    df, vectorizer = thesisModelFeatures.create_features_df(
      london_corpus = self.london_leftovers.filter_short_p(),
      zwickau_corpus = self.zwickau_leftovers.filter_short_p(),
      n_gram = n_gram,
      features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
      return_vectorizer = True,
      )
    self.london_vs_zwickau_features_df = df
    self.london_vs_zwickau_vectorizer = vectorizer
    return self.london_vs_zwickau_features_df
  
  def london_original_VS_zwickau_original(self, *, n_gram = (2, 5,), features = { 'tfidf', 'inner_mean_cosine_similarity_score' }):
    df, vectorizer = thesisModelFeatures.create_features_df(
      london_corpus = self.london.corpus,
      zwickau_corpus = self.zwickau.corpus,
      n_gram = n_gram,
      features = features,
      return_vectorizer = True,
      )
    self.london_original_vs_zwickau_original_features_df = df
    self.london_original_vs_zwickau_original_vectorizer = vectorizer
    return self.london_original_vs_zwickau_original_features_df

  def london_by_burchard_by_london_VS_zwickau_vectorizer(self):
    df = thesisModelFeatures.create_features_df(
      london_corpus = self.london_leftovers.filter_short_p(),
      n_gram = (2,5),
      features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
      vectorizer = self.burchard_by_london_VS_zwickau_vectorizer
      )
    self.london_by_burchard_by_london_VS_zwickau_vectorizer_features_df = df
    return self.london_by_burchard_by_london_VS_zwickau_vectorizer_features_df

  def zwickau_by_burchard_by_london_VS_london_vectorizer(self):
    df =  thesisModelFeatures.create_features_df(
      zwickau_corpus = self.zwickau_leftovers.filter_short_p(),
      n_gram = (2,5),
      features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
      vectorizer = self.burchard_by_london_VS_london_vectorizer
      )
    self.zwickau_by_burchard_by_london_VS_london_vectorizer_features_df = df
    return self.zwickau_by_burchard_by_london_VS_london_vectorizer_features_df

  def __burchard_VS_zwickau(self, burchard_corpus):
    return thesisModelFeatures.create_features_df(
      zwickau_corpus = self.zwickau_leftovers.filter_short_p(),
      burchard_corpus = burchard_corpus,
      n_gram = (2,5),
      features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
      return_vectorizer = True
      )

  def __burchard_VS_london(self, burchard_corpus):
    return thesisModelFeatures.create_features_df(
      london_corpus = self.london_leftovers.filter_short_p(),
      burchard_corpus = burchard_corpus,
      n_gram = (2,5),
      features = { 'tfidf', 'inner_mean_cosine_similarity_score' },
      return_vectorizer = True
      )