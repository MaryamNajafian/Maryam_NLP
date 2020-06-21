from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
Visualize the word-embedding from Glove
"""

from builtins import range
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main(we_file='glove_model_50.npz', w2i_file='glove_word2idx_50.json'):
    words = ['japan', 'japanese', 'england', 'english', 'australia', 'australian', 'china', 'chinese', 'italy', 'italian', 'french', 'france', 'spain', 'spanish']

    with open(w2i_file) as f: # load the word to index file
        word2idx = json.load(f)


    npz = np.load(we_file) # load the word embedding
    W = npz['arr_0']
    V = npz['arr_1']

    # we take average of the both output matrices (from Glove)
    # instead of visualizing them individually
    We = (W + V.T) / 2

    idx = [word2idx[w] for w in words]  #  word indexes of thee words that was selected
    # We = We[idx]

    tsne = TSNE()
    Z = tsne.fit_transform(We) # transforming the word embeddings
    Z = Z[idx] # indexing the word embedding
    plt.scatter(Z[:,0], Z[:,1]) # scatter plot with selected words
    for i in range(len(words)):
        plt.annotate(s=words[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


if __name__ == '__main__':
    main()
