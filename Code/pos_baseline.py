from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
* Implementing baseline Part of Speech (POS) tagging using: 
* Logistic Regression (LR) with stochastic gradient descent (SGD) in Theano is used
* INPUTs are one hot encoded words and OUTPUTs are tags
* If we use Scikit-learn version of logistic regress it is too slow 
  because it will try to  do a batch solution and there is too much data for that
* We compare the results with decision trees as a sanity check
* Note:
    for logistic regression we don't want to split the sequences
    we just want a set of data where its word that maps the tag
    VS with the RNN we want to split the sequences so that each
    separate sentence has a sequence and then we consider the prev.
    words to try to predict the current words tag
"""

from builtins import range

import os, sys
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import config

class LogisticRegression:
    """
    * calculate the accuracy
    * measure the F1 score
    """
    def __init__(self):
        pass

    def fit(self, X, Y, V=None, K=None, D=50, lr=1e-1, mu=0.99, batch_sz=100, epochs=6):


        # V: Vocab size
        # K: Number of classes
        # mu: Momentum
        if V is None:
            V = len(set(X))
        if K is None:
            K = len(set(Y))
        N = len(X)

        # Weight and bias initialization
        W = np.random.randn(V, K) / np.sqrt(V + K)
        b = np.zeros(K)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

        thX = T.ivector('X') # integer input vectors
        thY = T.ivector('Y') # integer output vectors

        py_x = T.nnet.softmax(self.W[thX] + self.b) # calculate p(y|x)
        prediction = T.argmax(py_x, axis=1) # each word will have its own unique tag

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params] # we need momentum

        # incase inputs are the wrong type we set allow_input_downcast to True
        self.cost_predict_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        updates = [
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]
        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True
        )

        costs = []
        n_batches = N // batch_sz
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            print("epoch:", i)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)
                if j % 200 == 0: #every 200 steps we print out the progress
                    print(
                        "i:", i, "j:", j,
                        "n_batches:", n_batches,
                        "cost:", c,
                        "error:", np.mean(p != Ybatch)
                    )
        plt.plot(costs)
        plt.show()

    def score(self, X, Y):
        _, p = self.cost_predict_op(X, Y)
        return np.mean(p == Y)

    def f1_score(self, X, Y):
        _, p = self.cost_predict_op(X, Y)
        return f1_score(Y, p, average=None).mean()


def get_data(split_sequences=False):
    train_text = config.CHUNKING_DATA + '/train.txt'
    test_text = config.CHUNKING_DATA + '/test.txt'

    if not os.path.exists('chunking'):
        print("Please create a folder in your local directory called 'chunking'")
        print("train.txt and test.txt should be stored in there.")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('chunking/train.txt'):
        print("train.txt is not in chunking/train.txt")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('chunking/test.txt'):
        print("test.txt is not in chunking/test.txt")
        print("Please check the comments to get the download link.")
        exit()

    """
    for logistic regression we don't want to split the sequences
    we just want a set of data where its word that maps the tag
    VS with the RNN we want to split the sequences so that each
    separate sentence has a sequence and then we consider the prev.
    words to try to predict the current words tag
    """
    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []

    for line in open(train_text):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])
            
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    # load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []
    for line in open(test_text):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx) # use this as unknown
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx

def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data()

    # convert to numpy arrays
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    # convert Xtrain to indicator matrix
    N = len(Xtrain)
    V = len(word2idx) + 1
    print("vocabulary size:", V)
    # Xtrain_indicator = np.zeros((N, V))
    # Xtrain_indicator[np.arange(N), Xtrain] = 1

    # decision tree
    dt = DecisionTreeClassifier()

    # without indicator
    dt.fit(Xtrain.reshape(N, 1), Ytrain)
    print("dt train score:", dt.score(Xtrain.reshape(N, 1), Ytrain))
    p = dt.predict(Xtrain.reshape(N, 1))
    print("dt train f1:", f1_score(Ytrain, p, average=None).mean())

    # with indicator -- too slow!!
    # dt.fit(Xtrain_indicator, Ytrain)
    # print("dt score:", dt.score(Xtrain_indicator, Ytrain))

    # train and score
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V)
    print("training complete")
    print("lr train score:", model.score(Xtrain, Ytrain))
    print("lr train f1:", model.f1_score(Xtrain, Ytrain))


    Ntest = len(Xtest)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    # convert Xtest to indicator
    # Xtest_indicator = np.zeros((Ntest, V))
    # Xtest_indicator[np.arange(Ntest), Xtest] = 1

    # decision tree test score
    print("dt test score:", dt.score(Xtest.reshape(Ntest, 1), Ytest))
    p = dt.predict(Xtest.reshape(Ntest, 1))
    print("dt test f1:", f1_score(Ytest, p, average=None).mean())
    # print("dt test score:", dt.score(Xtest_indicator, Ytest)) # too slow!

    # logistic test score -- too slow!!
    print("lr test score:", model.score(Xtest, Ytest))
    print("lr test f1:", model.f1_score(Xtest, Ytest))

if __name__ == '__main__':
    main()
