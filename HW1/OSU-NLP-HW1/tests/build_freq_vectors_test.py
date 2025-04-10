import unittest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Vocabulary import Vocabulary
from build_freq_vectors import compute_cooccurrence_matrix

class TestFreqVec(unittest.TestCase):
    def test_coocurrence_matrix(self):
        corpus = ["I love cats and dogs"]
        vocab = Vocabulary(corpus,min_freq = 1)

        C = compute_cooccurrence_matrix(corpus, vocab,window_size = 2)
        #print("\n",C)
        #print("\n")

        expected = np.array([
            [0, 1, 1, 0, 0, 0],  # I
            [1, 0, 1, 1, 0, 0],  # love
            [1, 1, 0, 1, 1, 0],  # cats
            [0, 1, 1, 0, 1, 0],  # and
            [0, 0, 1, 1, 0, 0],  # dogs
            [0, 0, 0, 0, 0, 0],  # UNK
        ])
        np.testing.assert_array_equal(C, expected, "Co-occurrence matrix does not match expected counts")

        corpus2 = ["a b a c a b"]
        vocab2 = Vocabulary(corpus2,min_freq = 1)

        C2 = compute_cooccurrence_matrix(corpus2, vocab2,window_size = 2)
        #print(C2)

        expected2 = np.array([
            [0, 3, 2, 0],  # a
            [3, 0, 2, 0],  # b
            [2, 2, 0, 0],  # c
            [0, 0, 0, 0],  # UNK
        ])

        np.testing.assert_array_equal(C2, expected2, "Matrix does not match expected values")



if __name__ == '__main__':
    unittest.main()