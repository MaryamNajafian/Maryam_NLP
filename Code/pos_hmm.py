from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
* Implementing Part of Speech (POS) tagging 
* INPUTs are one hot encoded words and OUTPUTs are tags
* Measure F1-score and accuracy
* Using HMMs and Viterbi algorithms to map from a sequence of words to a sequence of POS tags
    * Sequence of words : Observation
    * Sequence of POS tags: Hidden states
    * Sentences follow grammar rules and hence POS tags have a Markov structure 
"""

from builtins import range
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from hmmd_scaled import HMM
from pos_baseline import get_data
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score
sys.path.append(os.path.abspath('..'))

def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total


def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()


# def flatten(l):
#     return [item for sublist in l for item in sublist]


def main(smoothing=1e-1):
    # smooth state transition matrix and observation matrix
    # X = words, Y = POS tags
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
    V = len(word2idx) + 1

    # find hidden state transition matrix and pi
    # pi: initial state distribution
    # A: state transition matrix
    # B: output distribution
    # M: number of hidden states is equal to the number of POS tags

    M = max(max(y) for y in Ytrain) + 1 #len(set(flatten(Ytrain)))
    A = np.ones((M, M))*smoothing # add-one smoothing
    pi = np.zeros(M)


    for y in Ytrain:
        pi[y[0]] += 1 # y[0]s are start states
        for i in range(len(y)-1):
            A[y[i], y[i+1]] += 1 # y[i] and y[i+1] is the transition
    # turn it into a probability matrix
    A /= A.sum(axis=1, keepdims=True) # normalize all the distributions so they all add up to 1
    pi /= pi.sum()

    # find the observation matrix
    B = np.ones((M, V))*smoothing # add-one smoothing
    # the first dimension in  B is the state and the second is observation

    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1 # the state is the target so yi comes first
    B /= B.sum(axis=1, keepdims=True) # we normalize B

    hmm = HMM(M)
    # we set all HMM paramters to pi, A, and B
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # calculate the predictions
    Ptrain = []
    for x in Xtrain:
        p = hmm.get_state_sequence(x)
        Ptrain.append(p)

    Ptest = []
    for x in Xtest:
        p = hmm.get_state_sequence(x)
        Ptest.append(p)

    # print results
    print("train accuracy:", accuracy(Ytrain, Ptrain))
    print("test accuracy:", accuracy(Ytest, Ptest))
    print("train f1:", total_f1_score(Ytrain, Ptrain))
    print("test f1:", total_f1_score(Ytest, Ptest))

if __name__ == '__main__':
    main()
