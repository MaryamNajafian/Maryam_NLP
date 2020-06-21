from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
Use pretrained w2v for word analogy
word2vec has a vocab size of 3 million, D = 300
it also contains phrases New York , New_York, ...
word vectors are provided in a binary format, so we use gensim libary to access it as an object
"""

from future.utils import iteritems
from builtins import range
from gensim.models import KeyedVectors
import config


# %%

def find_analogies(w1, w2, w3, word_vectors):
    r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
    print("%s - %s = %s - %s" % (w1, w2, r[0][0], w3))


def nearest_neighbors(w, word_vectors):
    r = word_vectors.most_similar(positive=[w])
    print("neighbors of: %s" % w)
    for word, score in r:
        print("\t%s" % word)


def main():
    w2v_file = config.W2V_PRETRAINED
    word_vectors = KeyedVectors.load_word2vec_format(
        w2v_file,
        binary=True
    )

    # convenience
    # result looks like:
    # [('athens', 0.6001024842262268),
    #  ('albert', 0.5729557275772095),
    #  ('holmes', 0.569324254989624),
    #  ('donnie', 0.5690680742263794),
    #  ('italy', 0.5673537254333496),
    #  ('toni', 0.5666348338127136),
    #  ('spain', 0.5661854147911072),
    #  ('jh', 0.5661597847938538),
    #  ('pablo', 0.5631559491157532),
    #  ('malta', 0.5620371103286743)]

    find_analogies('king', 'man', 'woman', word_vectors)
    find_analogies('france', 'paris', 'london', word_vectors)
    find_analogies('france', 'paris', 'rome', word_vectors)
    find_analogies('paris', 'france', 'italy', word_vectors)
    find_analogies('france', 'french', 'english', word_vectors)
    find_analogies('japan', 'japanese', 'chinese', word_vectors)
    find_analogies('japan', 'japanese', 'italian', word_vectors)
    find_analogies('japan', 'japanese', 'australian', word_vectors)
    find_analogies('december', 'november', 'june', word_vectors)
    find_analogies('miami', 'florida', 'texas', word_vectors)
    find_analogies('einstein', 'scientist', 'painter', word_vectors)
    find_analogies('china', 'rice', 'bread', word_vectors)
    find_analogies('man', 'woman', 'she', word_vectors)
    find_analogies('man', 'woman', 'aunt', word_vectors)
    find_analogies('man', 'woman', 'sister', word_vectors)
    find_analogies('man', 'woman', 'wife', word_vectors)
    find_analogies('man', 'woman', 'actress', word_vectors)
    find_analogies('man', 'woman', 'mother', word_vectors)
    find_analogies('heir', 'heiress', 'princess', word_vectors)
    find_analogies('nephew', 'niece', 'aunt', word_vectors)
    find_analogies('france', 'paris', 'tokyo', word_vectors)
    find_analogies('france', 'paris', 'beijing', word_vectors)
    find_analogies('february', 'january', 'november', word_vectors)
    find_analogies('france', 'paris', 'rome', word_vectors)
    find_analogies('paris', 'france', 'italy', word_vectors)

    nearest_neighbors('king', word_vectors)
    nearest_neighbors('france', word_vectors)
    nearest_neighbors('japan', word_vectors)
    nearest_neighbors('einstein', word_vectors)
    nearest_neighbors('woman', word_vectors)
    nearest_neighbors('nephew', word_vectors)
    nearest_neighbors('february', word_vectors)
    nearest_neighbors('rome', word_vectors)


# %%
main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()
