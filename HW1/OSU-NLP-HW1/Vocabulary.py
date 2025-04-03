from collections import Counter 
from re import sub, compile
import re
import matplotlib.pyplot as plt
import numpy as np

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """
		text = text.lower()  # lowercase
		text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special characters
		tokens = text.split()  # tokenize
		return tokens



	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self,corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """
		word2idx = {}
		idx2word = {}
		freq = {}

		token_corpus = self.tokenize(corpus)

		for i, token in enumerate(token_corpus):
			if token not in word2idx:
				word2idx[token] = i
				idx2word[i] = token

			if token in freq:
				freq[token] += 1
			else:
				freq[token] = 1

		return word2idx, idx2word, freq
		# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		#raise UnimplementedFunctionError("You have not yet implemented build_vocab.")


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details
	    """ 
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		token_id, token_freq = zip(*freq)

		# Plot word frequency
		plt.figure(figsize=(10, 6))
		plt.plot(token_id, token_freq)
		plt.yscale('log')
		plt.xscale('log')
		plt.title(f"Token Frequency Distribution")
		plt.xlabel("Token ID (sorted by frequency")
		plt.ylabel("Frequency")
		plt.tight_layout()
		plt.show()

		total = sum(token_freq)
		cumulative = np.cumsum(token_freq)
		cumulative_fraction = cumulative / total

		# Plot cumulative coverage
		plt.figure(figsize=(10, 6))
		plt.plot(cumulative_fraction)
		plt.title("Cumulative Fraction Covered")
		plt.xlabel("Token ID (Sorted by Frequency)")
		plt.ylabel("Cumulative Fraction")
		plt.grid(True, linestyle='--', linewidth=0.5)
		plt.tight_layout()
		plt.show()
	    # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		#raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")
