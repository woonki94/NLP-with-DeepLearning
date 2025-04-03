import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Vocabulary import Vocabulary

class TestVocabulary(unittest.TestCase):
    def test_tokenize(self):
        vocab = Vocabulary("Some random String")
        string = "The blue dog jumped, but not high"
        token_list = ["the", "blue", "dog", "jumped", "but", "not", "high"]
        self.assertEqual(vocab.tokenize(string), token_list)

    def test_build_vocab(self):
        string = "The the the quick quick quick quick brown brown fox jumps over the lazy dog! The dog didn't mind."
        vocab = Vocabulary(string)
        word2idx, idx2word, freq = vocab.build_vocab(string)

        expected_tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'didnt', 'mind']
        expected_freq = {
            'the': 5,
            'quick': 4,
            'brown': 2,
            'fox': 1,
            'jumps': 1,
            'over': 1,
            'lazy': 1,
            'dog': 2,
            'didnt': 1,
            'mind': 1
        }
        print(word2idx,idx2word, freq)
        assert set(word2idx.keys()) == set(expected_tokens)
        assert set(idx2word.values()) == set(expected_tokens)
        assert freq == expected_freq

        for word, idx in word2idx.items():
            assert idx2word[idx] == word
        vocab.make_vocab_charts()

if __name__ == '__main__':
    unittest.main()
