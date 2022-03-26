import unittest
import utils.utils as thesisUtils

chunks_original_vs_expected = [
    [
        ["sentence 1.", "sentence 2."],
        [
            ["sentence 1.", "sentence 2."]
        ]
    ],
    [
        ["sentence 1.", "sentence 2.", "sentence 3.", "sentence 4.", "sentence 5.", "sentence 6."],
        [
            ["sentence 1.", "sentence 2.", "sentence 3."], 
            ["sentence 4.", "sentence 5.", "sentence 6."]
        ]
    ],
    [
        ["sentence 1.", "sentence 2.", "sentence 3.", "sentence 4.", "sentence 5."],
        [
            ["sentence 1.", "sentence 2.", "sentence 3.", "sentence 4.", "sentence 5."]
        ]
    ],
    [
        ["sentence 1.", "sentence 2.", "sentence 3.", "sentence 4.", "sentence 5.", "sentence 6.", "sentence 7.", "sentence 8."],
        [
            ["sentence 1.", "sentence 2.", "sentence 3."],
            ["sentence 4.", "sentence 5.", "sentence 6.", "sentence 7.", "sentence 8."],
        ]
    ],
    [
        ["sentence 1.", "sentence 2.", "sentence 3.", "sentence 4.", "sentence 5.", "sentence 6.", "sentence 7.", "sentence 8.", "sentence 9."],
        [
            ["sentence 1.", "sentence 2.", "sentence 3."],
            ["sentence 4.", "sentence 5.", "sentence 6."],
            ["sentence 7.", "sentence 8.", "sentence 9."]
        ]
    ],
    [
        ["sentence 1.", "sentence 2.", "sentence 3.", "sentence 4.", "sentence 5.", "sentence 6.", "sentence 7.", "sentence 8.", "sentence 9.", "sentence 10."],
        [
            ["sentence 1.", "sentence 2.", "sentence 3."],
            ["sentence 4.", "sentence 5.", "sentence 6."],
            ["sentence 7.", "sentence 8.", "sentence 9.", "sentence 10."]
        ]
    ]
]

class TestChunkMethod(unittest.TestCase):

    def test_chunk(self):
        for original, expected in chunks_original_vs_expected:
            self.assertEqual(list(thesisUtils.chunks(original, 3)), expected)
        

if __name__ == '__main__':
    unittest.main()