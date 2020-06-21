from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'


"""
    Input:
        load in the brown corpus data: brown corpus comprises list of sentences
        load a dict (mapping of words to indices) 
        load numeric sentence representations  (it's a sequences of ints instead of strings) 
        load raw bigram probability matrix
    Output:
        find bigram probabilities using logistic regression 
        compare them with the bigram probabilities found using counts
        use plots to show that logistic regression weights are equivalent to probabilities derived from counting
"""

from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import os
import sys

sys.path.append(os.path.abspath('..'))

from rnn_c.util import get_wikipedia_data
from rnn_c.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
from bigram_prob_estimate_with_markov import get_bigram_probs


def main():
    # load in the data
    # note: sentences are already converted to sequences of word indexes
    # note: you can limit the vocab size if you run out of memory
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
    # sentences, word2idx = get_sentences_with_word2idx()

    # vocab size
    V = len(word2idx)
    print("Vocab size:", V)

    # we will also treat beginning of sentence and end of sentence as bigrams
    # START -> first word
    # last word -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    # train a logistic model
    # W is logistic regression weights
    # initialize a random weight matrix
    W = np.random.randn(V, V) / np.sqrt(V)
    losses = [] # store loss per iteration
    epochs = 1  # we got convergence after 1 epoch as sentences follow similar patterns
    lr = 1e-1   # learning rate

    def softmax(a):
        a = a - a.max() # subtract the max to eliminate numerical overflow
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    # what is the loss if we set W = log(bigram_probs)?
    W_bigram = np.log(bigram_probs)
    bigram_losses = []

    t0 = datetime.now()
    for epoch in range(epochs):
        # shuffle sentences at each epoch
        random.shuffle(sentences)

        j = 0  # keep track of iterations
        # p(Y|X) = prediction = softmax(inputs.W^T)
        # goal: minimize predictions - targets
        for sentence in sentences:
            # convert sentence into one-hot encoded inputs and targets
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = np.zeros((n - 1, V))
            targets = np.zeros((n - 1, V))
            inputs[np.arange(n - 1), sentence[:n - 1]] = 1
            targets[np.arange(n - 1), sentence[1:]] = 1

            # get output predictions
            predictions = softmax(inputs.dot(W))

            # do a gradient descent step
            # for training the model ONLY gradient of loss is needed not the loss
            gradient_of_loss = inputs.T.dot(predictions - targets)
            W = W - lr * gradient_of_loss

            # keep track of the loss ONLY for debugging purposes and plotting
            loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            losses.append(loss)

            # keep track of the bigram loss
            # only do it for the first epoch to avoid redundancy
            if epoch == 0:
                bigram_predictions = softmax(inputs.dot(W_bigram))
                bigram_loss = -np.sum(targets * np.log(bigram_predictions)) / (n - 1)
                bigram_losses.append(bigram_loss)

            # to ensure cost is decreasing at every step we print the loss
            if j % 10 == 0:
                print("epoch:", epoch, "sentence: %s/%s" % (j, len(sentences)), "loss:", loss)
            j += 1

    print("Elapsed time training:", datetime.now() - t0) # print time spent
    plt.plot(losses) # plot loss per iteration

    # plot a horizontal line for the bigram loss
    avg_bigram_loss = np.mean(bigram_losses)
    print("avg_bigram_loss:", avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')

    # plot smoothed losses to reduce variability
    def smoothed_loss(x, decay=0.99):
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y

    plt.plot(smoothed_loss(losses))
    plt.show()

    # for the most common 200 words
    # plot logistic regression weights W (log of prob estimated by logistic regression) and bigram count probability side-by-side

    plt.subplot(1, 2, 1)
    plt.title("Logistic Model")
    plt.imshow(softmax(W))
    plt.subplot(1, 2, 2)
    plt.title("Bigram Probs")
    plt.imshow(bigram_probs)
    plt.show()


# %%
main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()
