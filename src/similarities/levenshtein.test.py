import unittest
import similarities.levenshtein as levenshtein_similarity

class TestTextCleanupMethods(unittest.TestCase):

    def test_get_similarity_tupels(self):
      self.maxDiff = None
      words_1 = ['hello', 'hello world']
      words_2 = ['hello', 'helo world']
      result = levenshtein_similarity.get_similarity_tupels(words_1, words_2)

      self.assertEqual(result, [('hello', 'hello', 0), ('hello world', 'helo world', 1)])

    def test_get_sentence_score(self):
      self.maxDiff = None
      sentence_1 = 'hello world'
      sentence_2 = 'helo world worl'

      self.assertEqual(
        levenshtein_similarity.get_sentence_score(sentence_1, sentence_2), 
        1
        )

      self.assertEqual(
        levenshtein_similarity.get_sentence_score(sentence_2, sentence_1), 
        2
        )

if __name__ == '__main__':
  unittest.main()