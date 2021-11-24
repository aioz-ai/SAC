import numpy as np
import _pickle as pickle
from math import *

def convert_index_to_word(cls_ids, classes):
    words = []
    for ids in cls_ids:
        b_words = []
        for idx in ids:
            cls = classes[idx]
            b_words.append(cls)
        words.append(b_words)
    words = np.array(words)
    return words


def read_glove_vecs(glove_file, dictionary_file):
    d = pickle.load(open(dictionary_file, 'rb'))
    word_to_vec_map = np.load(glove_file)
    words_to_index = d[0]
    index_to_words = d[1]
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = (X[i].lower()).split()
        sentence_words = sentence_words[:max_len]
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1
    return X_indices


def cosine_annealing(step, n_iters, n_cycles, lrate_max):
    iter_per_cycle = n_iters / n_cycles
    cos_inner = (pi * (step % iter_per_cycle)) / (iter_per_cycle)
    lr = lrate_max / 2 * (cos(cos_inner) + 1)
    return np.array(lr).astype(np.float32)


def num_params(variables):
    total_params = 0
    for v in variables:
        shape = v.get_shape()
        params = 1
        for dim in shape:
            params *= dim
            total_params += params
    return total_params
