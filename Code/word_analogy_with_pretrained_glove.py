from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
Use pretrained Glove for word analogy
GloVe has a vocab size of 400k, D = 300
word vectors are provided in a text format
"""

from future.utils import iteritems
from builtins import range
import config
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

#%%

def load_pre_trained_glove_wordvectors():
    """
    load in pre-trained word vectors

    Inputs: Read the text file containing GloVe word  embeddings
    GloVe: https://nlp.stanford.edu/projects/glove/
    Direct link: http://nlp.stanford.edu/data/glove.6B.zip

    Outputs: create word2vec embeddings
    """
    print('Loading word vectors...')
    glove_path = config.GLOVE_PRETRAINED
    word2vec = {}
    embedding = []
    idx2word = []

    with open(glove_path, encoding='utf-8') as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
            embedding.append(vec)
            idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))
    embedding = np.array(embedding)
    V, D = embedding.shape
    return word2vec, embedding, idx2word, V, D


def dist1(a, b):
    return np.linalg.norm(a - b)


def dist2(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_analogies_not_efficient(w1, w2, w3, word2vec, embedding, idx2word, V, D):
    # more intuitive
    pick = 1  # pick a distance type

    if pick == 1:
        dist, metric = dist2, 'cosine'
    elif pick == 2:
        dist, metric = dist1, 'euclidean'

    for w in (w1, w2, w3):
        if w not in word2vec:
            print("%s not in dictionary" % w)
            return

    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman

    min_dist = float('inf')
    best_word = ''
    for word, v1 in iteritems(word2vec):
        if word not in (w1, w2, w3):
            d = dist(v0, v1)
            if d < min_dist:
                min_dist = d
                best_word = word
    print(w1, "-", w2, "=", best_word, "-", w3)


def find_analogies(w1, w2, w3, word2vec, embedding, idx2word, V, D):
    ## faster
    pick = 1  # pick a distance type

    if pick == 1:
        metric = 'cosine'
    elif pick == 2:
        metric = 'euclidean'

    for w in (w1, w2, w3):
        if w not in word2vec:
            print("%s not in dictionary" % w)
            return

    for w in (w1, w2, w3):
        if w not in word2vec:
            print("%s not in dictionary" % w)
            return

    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman

    distances = pairwise_distances(v0.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[:4]
    for idx in idxs:
        word = idx2word[idx]
        if word not in (w1, w2, w3):
            best_word = word
            break

    print(w1, "-", w2, "=", best_word, "-", w3)


def nearest_neighbors(w, n, word2vec, embedding, idx2word, V, D):
    pick = 1  # pick a distance type

    if pick == 1:
        metric = 'cosine'
    elif pick == 2:
        metric = 'euclidean'
    if w not in word2vec:
        print("%s not in dictionary:" % w)
        return

    v = word2vec[w]
    """ 
    shape of A, B, dist(A,B) N1xD, N2xD, N1xN2 
    """

    distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[1:n + 1]
    print("neighbors of: %s" % w)
    for idx in idxs:
        print("\t%s" % idx2word[idx])


# %%
def main():
    word2vec, embedding, idx2word, V, D = load_pre_trained_glove_wordvectors()

    find_analogies('king', 'man', 'woman', word2vec)
    find_analogies('france', 'paris', 'london', word2vec, embedding, idx2word, V, D)
    find_analogies('france', 'paris', 'rome', word2vec, embedding, idx2word, V, D)
    find_analogies('paris', 'france', 'italy', word2vec, embedding, idx2word, V, D)
    find_analogies('france', 'french', 'english', word2vec, embedding, idx2word, V, D)
    find_analogies('japan', 'japanese', 'chinese', word2vec, embedding, idx2word, V, D)
    find_analogies('japan', 'japanese', 'italian', word2vec, embedding, idx2word, V, D)
    find_analogies('japan', 'japanese', 'australian', word2vec, embedding, idx2word, V, D)
    find_analogies('december', 'november', 'june', word2vec, embedding, idx2word, V, D)
    find_analogies('miami', 'florida', 'texas', word2vec, embedding, idx2word, V, D)
    find_analogies('einstein', 'scientist', 'painter', word2vec, embedding, idx2word, V, D)
    find_analogies('china', 'rice', 'bread', word2vec, embedding, idx2word, V, D)
    find_analogies('man', 'woman', 'she', word2vec, embedding, idx2word, V, D)
    find_analogies('man', 'woman', 'aunt', word2vec, embedding, idx2word, V, D)
    find_analogies('man', 'woman', 'sister', word2vec, embedding, idx2word, V, D)
    find_analogies('man', 'woman', 'wife', word2vec, embedding, idx2word, V, D)
    find_analogies('man', 'woman', 'actress', word2vec, embedding, idx2word, V, D)
    find_analogies('man', 'woman', 'mother', word2vec, embedding, idx2word, V, D)
    find_analogies('heir', 'heiress', 'princess', word2vec, embedding, idx2word, V, D)
    find_analogies('nephew', 'niece', 'aunt', word2vec, embedding, idx2word, V, D)
    find_analogies('france', 'paris', 'tokyo', word2vec, embedding, idx2word, V, D)
    find_analogies('france', 'paris', 'beijing', word2vec, embedding, idx2word, V, D)
    find_analogies('february', 'january', 'november', word2vec, embedding, idx2word, V, D)
    find_analogies('france', 'paris', 'rome', word2vec, embedding, idx2word, V, D)
    find_analogies('paris', 'france', 'italy', word2vec, embedding, idx2word, V, D)

    nearest_neighbors('king', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('france', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('japan', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('einstein', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('woman', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('nephew', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('february', 5, word2vec, embedding, idx2word, V, D)
    nearest_neighbors('rome', 5, word2vec, embedding, idx2word, V, D)


# %%
main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()
