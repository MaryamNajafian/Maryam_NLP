from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

from builtins import range
from nltk.corpus import brown
import operator

"""
    Input: 
            load in the brown corpus data: brown corpus comprises list of sentences
    Output: 
            Return a dict (mapping of words to indices)  
            Return numeric sentence representation using a sequences of ints instead of strings
    Process:    
        * convert list of strings in sentences to a sequences of integers 
        * assign a unique int to every word that appears in corpus
        * create a dictionary: contains a mapping from a word to it's corresponding index
        * we indicate beginning and end of every sentences using START and End tokens 

"""
# A set of words that we want to make sure is part of the dictionary
KEEP_WORDS = set([
    'king', 'man', 'queen', 'woman',
    'italy', 'rome', 'france', 'paris',
    'london', 'britain', 'england',
])


def get_sentences():
    # returns Brown corpus sentences, comprises list of sentences
    return brown.sents()


def get_sentences_with_word2idx():

    sentences = get_sentences() # load in the brown corpus data, comprises list of sentences
    indexed_sentences = []

    i = 2
    word2idx = {'START': 0, 'END': 1}
    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2idx:
                word2idx[token] = i
                i += 1

            indexed_sentence.append(word2idx[token])
        indexed_sentences.append(indexed_sentence)

    print("Vocab size:", i)
    return indexed_sentences, word2idx

    # load in the brown corpus data
    # brown corpus comprises list of sentences
    # now we convert list of strings in sentences to a sequences of word indices

def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
    """
    load in the brown corpus data
    brown corpus comprises list of sentences
    now we convert list of strings in sentences to a sequences of word indices
    here the vocab size is limited  (use if: run out of memory or faster results)
    """
    sentences = get_sentences()
    indexed_sentences = []

    i = 2
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']

    # set count of START & END (at index:0 & 1) to inf
    # so they won't get filtered when we pick most frequent vocab
    word_idx_count = {
        0: float('inf'),
        1: float('inf'),
    }

    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = i
                i += 1
            # keep track of counts for later sorting
            idx = word2idx[token]
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1  # if value of that key in dict didn't exist it is set to
            # 0 then added by 1

            indexed_sentence.append(idx)
        indexed_sentences.append(indexed_sentence)

    # restrict vocab size

    # set all the words I want to keep to infinity
    # so that they are included when I pick the most
    # common words
    for word in keep_words:
        word_idx_count[word2idx[word]] = float('inf')

    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx

    assert ('START' in word2idx_small)
    assert ('END' in word2idx_small)
    for word in keep_words:
        assert (word in word2idx_small)

    # map old idx to new idx
    sentences_small = []
    for sentence in indexed_sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small
