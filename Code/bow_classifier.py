from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
1-Load REUTERS train and test data
2-Convert the data into GloVe or w2v vectors
3- Build a class with a scikitlearn like interface: fit/transfor/fit_transform
4-Transform the data,train a classifier, print the train and test accuracies
"""

from builtins import range
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors


class GloveVectorizer:
    def __init__(self):
        # load in pre-trained word vectors
        print('Loading word vectors...')
        word2vec = {}
        embedding = []
        idx2word = []
        with open(config.GLOVE_PRETRAINED) as f:
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

        # save for later
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v: k for k, v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class Word2VecVectorizer:
    def __init__(self):

        print("Loading in word vectors...")

        self.word_vectors = KeyedVectors.load_word2vec_format(
            config.W2V_PRETRAINED,
            binary=True
        )
        print("Finished loading in word vectors")

    def fit(self, data):
        pass

    def transform(self, data):
        # determine the dimensionality of vectors
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:

            tokens = sentence.split()  # we dont lower case the tokens
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found in w2v
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def main():
    # data from https://www.cs.umb.edu/~smimarog/textmining/datasets/
    train = pd.read_csv(config.REUTERS_TRAIN, header=None, sep='\t')
    test = pd.read_csv(config.REUTERS_TEST, header=None, sep='\t')
    train.columns = ['label', 'content']
    test.columns = ['label', 'content']
    option = 1
    if option == 1:
        vectorizer = GloveVectorizer()
    else:
        vectorizer = Word2VecVectorizer()

    Xtrain = vectorizer.fit_transform(train.content)
    Ytrain = train.label

    Xtest = vectorizer.transform(test.content)
    Ytest = test.label

    # create the model, train it, print scores
    model = RandomForestClassifier(n_estimators=200)
    model.fit(Xtrain, Ytrain)
    print("train score:", model.score(Xtrain, Ytrain))
    print("test score:", model.score(Xtest, Ytest))


# %%
main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()
