
from datasets import load_dataset, DatasetDict, load_from_disk
from patsy.state import center
from sklearn.cluster import KMeans

from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE
import os
import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab, window_size=5):
	"""

	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns:
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

	    """
	N = vocab.size
	C = np.zeros((N, N), dtype=np.int32)

	for sentence in corpus:
		tokens = vocab.tokenize(sentence)
		token_indices = [vocab.word2idx.get(token, vocab.word2idx['UNK']) for token in tokens]

		for i, center_id in enumerate(token_indices):
			start = max(0, i - window_size)
			end = min(len(token_indices), i + window_size + 1)

			for j in range(start, end):
				if i == j:
					continue  # Skip center word
				context_id = token_indices[j]

				if context_id == center_id:
					continue  # Skip self-cooccurrence

				C[center_id, context_id] += 1

	return C



	# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
	#raise UnimplementedFunctionError("You have not yet implemented compute_count_matrix.")


###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	"""

	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function.

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns:
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	    """
	C = compute_cooccurrence_matrix(corpus, vocab) # shape: (N, N)

	total = np.sum(C)  # total co-occurrence count
	row_sums = np.sum(C, axis=1, keepdims=True)  # shape: (N, 1)
	col_sums = np.sum(C, axis=0, keepdims=True)  # shape: (1, N)

	# Prevent divide-by-zero
	row_sums[row_sums == 0] = 1e-8
	col_sums[col_sums == 0] = 1e-8
	joint_prob = C / total
	marginal_prob_row = row_sums / total
	marginal_prob_col = col_sums / total

	expected = np.dot(marginal_prob_row, marginal_prob_col)

	pmi = np.log(joint_prob/(expected +1e-8)+1e-8)
	ppmi = np.maximum(pmi, 0)

	return ppmi
	# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
	#raise UnimplementedFunctionError("You have not yet implemented compute_ppmi_matrix.")




################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():
	DATASET_DIR = "../dataset/"

	# Check if dataset is already stored
	if os.path.exists(DATASET_DIR):
		logging.info("Loading dataset")
		dataset = load_from_disk(DATASET_DIR)
	else:
		logging.info("Downloading dataset")
		dataset = load_dataset("ag_news")
		dataset.save_to_disk(DATASET_DIR)

	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]

	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	logging.info("Building vocabulary Done")
	vocab.make_vocab_charts()

	logging.info("Plotting Done")
	plt.close()
	plt.pause(0.01)


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)

	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)
	plot_zoomed_tsne(word_vectors, vocab,'tsne_zoomed01', xlim=(15, 25), ylim=(55, 65))
	plot_zoomed_tsne(word_vectors, vocab, 'tsne_zoomed01', xlim=(-60,-50), ylim=(-5,5))

def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1)
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab, filename = 'tsne.png'):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))

	plt.figure(figsize=(16, 10), dpi=300)
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)

	#plt.show()
	#plt.savefig("./plots/tsne.png", dpi=300)
	output_path = os.path.join('./plots', filename)
	plt.savefig(output_path, dpi=300)


def plot_zoomed_tsne(word_vectors, vocab,filename, xlim=(-80, -60), ylim=(-10, 10)):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))

	# Filter for points inside the zoom window
	zoomed_idx = [i for i in top_word_idx if xlim[0] <= coords[i,0] <= xlim[1] and ylim[0] <= coords[i,1] <= ylim[1]]

	plt.figure(figsize=(16, 10), dpi=300)
	plt.plot(coords[zoomed_idx, 0], coords[zoomed_idx, 1], 'o', markerfacecolor='none',
	         markeredgecolor='k', alpha=0.6, markersize=4)

	for i in zoomed_idx:
		plt.annotate(vocab.idx2text([i])[0],
		             xy=(coords[i, 0], coords[i, 1]),
		             xytext=(5, 2),
		             textcoords='offset points',
		             ha='right',
		             va='bottom',
		             fontsize=5)

	plt.title("Zoomed-In t-SNE View")
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.tight_layout()
	output_path = os.path.join('./plots', filename)
	plt.savefig(output_path, dpi=300)
	#plt.show()

if __name__ == "__main__":
    main_freq()

