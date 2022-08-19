import unittest
import data.reader as dataReader
import similarities.cosine as thesisCosineSimilarity

class TestCosineSimilarities(unittest.TestCase):
  def test_london_zwickau_indexes_of_similarities_greater_than_08(self):
    london_corpus = dataReader.CorpusByNewLine.london()
    zwickau_corpus = dataReader.CorpusByNewLine.zwickau()

    london_zwickau_similarities = thesisCosineSimilarity.CrossVersionSimilarity5Gram(london_corpus, zwickau_corpus)
    london_zwickau_similarities.calculate()
    result = set(london_zwickau_similarities.get_matches_higher_than(0.8).original_indexes())
    expected = set([1, 2, 3, 4, 5, 8, 40, 48, 117, 119, 166, 169, 185, 192, 193, 237, 238, 239, 241, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 260, 261, 262, 263, 264, 265, 266, 268, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 304, 305, 306, 307, 308, 309, 310, 311])
    self.assertEqual(result, expected)

  def test_zwickau_london_indexes_of_similarities_greater_than_08(self):
    london_corpus = dataReader.CorpusByNewLine.london()
    zwickau_corpus = dataReader.CorpusByNewLine.zwickau()

    zwickau_london_similarities = thesisCosineSimilarity.CrossVersionSimilarity5Gram(zwickau_corpus, london_corpus)
    zwickau_london_similarities.calculate()
    result = set(zwickau_london_similarities.get_matches_higher_than(0.8).original_indexes())
    expected = set([1, 2, 3, 4, 5, 8, 40, 51, 122, 124, 169, 170, 176, 179, 233, 234, 235, 237, 239, 240, 241, 242, 243, 244, 245, 247, 248, 249, 250, 254, 268, 269, 271, 272, 273, 274, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 312, 313, 314, 315, 316, 317, 318, 319])
    self.assertEqual(result, expected)

if __name__ == '__main__':
  unittest.main()