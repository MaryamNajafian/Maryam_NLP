from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
    Input:
        load in the brown corpus data: brown corpus comprises list of sentences
        load a dict (mapping of words to indices) 
        load numeric sentence representations  (it's a sequences of ints instead of strings)  
    Output:
        find raw bigram probabilities for sentences
        find normalized log prob score per sentence so we can compare the log likelihood of real/fake sentences 
        scores random sentences using log likelihood of sentences 
    Process:
        Given one of the following sentences the system scores the sentence 
        (a) a random sentence is picked from the corpus or 
        (b) or a sentence is generated using random words from corpus 
        (c) or user types a sentence        
"""

from future.utils import iteritems
from builtins import range, input
import numpy as np
import os
import sys
import config
from rnn_c.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

#%%
def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):

    """
        structure of bigram probability matrix will be:
        (last word, current word) --> probability
        we will use add-1 smoothing
        note: we'll always ignore this from the END token
        * sentence[i] value indicates row/col index in VxV dimensional bigram_probs matrix
        * fill in VxV dimensional bigram_probs matrix: by calculating occurrences
          of (prev word (row index) | current word (col index)) for every sentence
    """
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                # beginning word
                bigram_probs[start_idx, sentence[i]] += 1
            else:
                # middle word
                bigram_probs[sentence[i - 1], sentence[i]] += 1

            # if we're at the final word
            # we update the bigram for last -> current
            # AND current -> END token
            if i == len(sentence) - 1:
                # final word
                bigram_probs[sentence[i], end_idx] += 1

    # normalize the counts along the rows to get probabilities
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs


# %%
# a function to calculate normalized log prob score
# for a sentence
def get_score(sentence, bigram_probs, start_idx, end_idx):
    score = 0
    for i in range(len(sentence)):
        if i == 0:
            # beginning word
            score += np.log(bigram_probs[start_idx, sentence[i]])
        else:
            # middle word
            score += np.log(bigram_probs[sentence[i - 1], sentence[i]])
    # final word
    score += np.log(bigram_probs[sentence[-1], end_idx])

    # normalize the score
    return score / (len(sentence) + 1)


# %%
# a function to map word indexes back to real words
def get_words(sentence, idx2word):
    return ' '.join(idx2word[i] for i in sentence)


# %%

def main():
    # load in the brown corpus data
    # brown corpus comprises list of sentences
    # now we convert list of strings in sentences to a sequences of word indices
    # you can limit the vocab size if you run out of memory
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(10000)
    # sentences, word2idx = get_sentences_with_word2idx()

    # vocab size
    V = len(word2idx)
    print("Vocab size:", V)

    # we will also treat beginning of sentence and end of sentence as bigrams
    # START -> followed by first word
    # last word followed by -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    idx2word = dict((v, k) for k, v in iteritems(word2idx))

    # when we sample a fake sentence, we want to ensure not to sample
    # start token or end token
    sample_probs = np.ones(V)
    sample_probs[start_idx] = 0
    sample_probs[end_idx] = 0
    sample_probs /= sample_probs.sum()

    # test our model on real and fake sentences
    while True:
        # real sentence
        real_idx = np.random.choice(len(sentences))
        real = sentences[real_idx]

        # fake sentence
        fake = np.random.choice(V, size=len(real), p=sample_probs)

        print("REAL:", get_words(real, idx2word), "SCORE:", get_score(real, bigram_probs, start_idx, end_idx))
        print("FAKE:", get_words(fake, idx2word), "SCORE:", get_score(fake, bigram_probs, start_idx, end_idx))

        # input your own sentence
        custom = input("Enter your own sentence:\n")
        custom = custom.lower().split()

        # check that all tokens exist in word2idx (otherwise, we can't get score)
        bad_sentence = False
        for token in custom:
            if token not in word2idx:
                bad_sentence = True

        if bad_sentence:
            print("Sorry, you entered words that are not in the vocabulary")
        else:
            # convert sentence into list of indexes
            custom = [word2idx[token] for token in custom]
            print("SCORE:", get_score(custom, bigram_probs, start_idx, end_idx))

        cont = input("Continue? [Y/n]")
        if cont and cont.lower() in ('N', 'n'):
            break


# %%
main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()
# %%
