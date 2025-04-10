from collections import Counter
from re import sub, compile
import re
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus,min_freq=30):

		self.min_freq=min_freq
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

		lemmatizer = WordNetLemmatizer()
		text = text.lower()
		text = re.sub(r'[^a-z0-9\s]', '', text)
		tokens = text.split()
		tokens = [lemmatizer.lemmatize(token) for token in tokens]
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

		joined_corpus = " ".join(corpus)
		token_corpus = self.tokenize(joined_corpus)

		# Count frequencies first
		freq = dict(Counter(token_corpus))

		# Rebuild vocab with cutoff
		index = 0
		unk_count = 0
		for token in freq:
			if freq[token] >= self.min_freq:
				word2idx[token] = index
				idx2word[index] = token
				index += 1
			else:
				unk_count += freq[token]

		# Add UNK token
		word2idx['UNK'] = index
		idx2word[index] = 'UNK'
		freq['UNK'] = unk_count

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
		plt.plot(range(len(token_freq)), token_freq)
		plt.yscale('log')

		plt.axhline(y=self.min_freq, color='red', linestyle='-', label=f'freq = {self.min_freq}')
		plt.text(len(token_freq) * 0.7, self.min_freq, f'freq = {self.min_freq}', color='red', va='bottom', ha='left')

		#step = max(1, len(token_freq) // 10)
		step = 10000  # fixed interval
		xticks = np.arange(0, len(token_freq), step)
		plt.xticks(xticks)

		plt.title("Token Frequency Distribution")
		plt.xlabel("Token ID (sorted by frequency)")
		plt.ylabel("Frequency")
		plt.tight_layout()
		plt.savefig("./plots/token_frequency_distribution.png", dpi=600)

		# Convert to numpy for vectorized ops
		token_freq = np.array(token_freq)
		total = np.sum(token_freq)

		# Find tokens that meet the min_freq threshold
		mask = token_freq >= self.min_freq
		included_freq = token_freq[mask]
		included_count = len(included_freq)
		covered_fraction = np.sum(included_freq) / total

		total = sum(token_freq)
		cumulative = np.cumsum(token_freq)
		cumulative_fraction = cumulative / total

		# Plot cumulative coverage
		plt.figure()
		plt.plot(cumulative_fraction)


		plt.axvline(x=included_count, color='red', linestyle='-')
		label_text = f"{covered_fraction:.3f} covered"
		plt.text(included_count + len(token_freq) * 0.01, 0.05, label_text,
				 color='red', rotation=90, va='bottom')

		plt.title("Cumulative Fraction Covered")
		plt.xlabel("Token ID (Sorted by Frequency)")
		plt.ylabel("Cumulative Fraction")
		plt.grid(True, linestyle='--', linewidth=0.5)
		plt.tight_layout()
		#plt.show()
		plt.savefig("./plots/cumulative_fraction_covered.png", dpi=600)

	    # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		#raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")
