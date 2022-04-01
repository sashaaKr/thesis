import unittest
import vocabulary.vocabulary as thesisVocabulary

vocabulary_for_2_versions_original_vs_expected = [
    [
        ['hello world hello'], ['world peace world world'], ['cat dog'],
        { 'world': { 'zwickau': 1, 'london': 3 } }
    ],

    [
        ['hello hello world'], ['hello hello world world world peace peace'], ['cat dog world'],
        { 'hello': { 'zwickau': 2, 'london': 2 } }
    ]
]

class TestVocabularyMethods(unittest.TestCase):

    def test_get_shared_vocabulary_for_2_versions(self):
        self.maxDiff = None

        for zwickau_vocab, london_vocab, breslau_vocab, expected in vocabulary_for_2_versions_original_vs_expected:
            result = thesisVocabulary.get_shared_vocabulary_for_2_versions(
                zwickau_vocab,
                'zwickau',
                london_vocab,
                'london',
                breslau_vocab,
            )
            self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()