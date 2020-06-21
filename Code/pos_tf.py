from __future__ import barry_as_FLUFL, print_function, division

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
* Implementing Part of Speech (POS) tagging 
* Using RNN in Tensorflow

   structure: Embedding --> GRU --> Dense

* INPUTs are one hot encoded words and OUTPUTs are tags
* Measure F1-score and accuracy
* Note: in TF (unlike Theano) all sequences should have equal length
  Anything that is shorter than the longest sequence is 0 padded
  You can think of your data as a NXTXD 
  * N samples
  * samples of length T
  * D is the dimensionality of each word vector
  This allows us to process our data in batches 
  which is more difficult in Theano where you are 
  going to have variable length sequences  
"""

from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

sys.path.append(os.path.abspath('..'))
from pos_baseline import get_data
from sklearn.utils import shuffle
from util import init_weight
from datetime import datetime
from sklearn.metrics import f1_score
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
import config


def get_data(split_sequences=False):
    train_text = config.CHUNKING_DATA + '/train.txt'
    test_text = config.CHUNKING_DATA + '/test.txt'
    if not os.path.exists(config.CHUNKING_DATA):
        print("Please create a folder in your local directory called 'chunking'")
        print("train.txt and test.txt should be stored in there.")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists(train_text):
        print("train.txt is not in chunking/train.txt")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists(test_text):
        print("test.txt is not in chunking/test.txt")
        print("Please check the comments to get the download link.")
        exit()

    word2idx = {}
    tag2idx = {}
    # unlike the Theano version of this code the index starts
    # from 1 because TF needs all input seq. to be the same size
    # and it does 0 padding and 0 is a special number that we can't
    # use for anything else
    word_idx = 1
    tag_idx = 1

    # X/Ytrain:each element is a sample and each sample is a list containing word or tag indeces
    Xtrain = []
    Ytrain = []

    # currentX/Y: contain a list of words and tags in the current sentence
    currentX = []
    currentY = []

    # each line contains one word and one tag
    # each sentence is separated by a blank line
    for line in open(train_text):
        line = line.rstrip()
        if line:  # check if the line is blank
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
        elif split_sequences:  # add the whole list as a sample
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
                currentX.append(word_idx)  # use this as unknown
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


def flatten(l):  # we need to flatten our data which is a list of lists
    return [item for sublist in l for item in sublist]


# get the data
Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
V = len(word2idx) + 2  # vocab size (+1 for including an index for unknown, +1 for starting from 1 rather than 0)
K = len(set(flatten(Ytrain)) | set(
    flatten(Ytest))) + 1  # num classes (assumption no unknown index, +1 for starting from 1 rather than 0)

# training config
epochs = 20
learning_rate = 1e-2
mu = 0.99
batch_size = 32
hidden_layer_size = 10
embedding_dim = 10
sequence_length = max(len(x) for x in Xtrain + Xtest)

# pad sequences
Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain, maxlen=sequence_length)
Ytrain = tf.keras.preprocessing.sequence.pad_sequences(Ytrain, maxlen=sequence_length)
Xtest = tf.keras.preprocessing.sequence.pad_sequences(Xtest, maxlen=sequence_length)
Ytest = tf.keras.preprocessing.sequence.pad_sequences(Ytest, maxlen=sequence_length)
print("Xtrain.shape:", Xtrain.shape)
print("Ytrain.shape:", Ytrain.shape)

# inputs
inputs = tf.placeholder(tf.int32, shape=(None, sequence_length))
targets = tf.placeholder(tf.int32, shape=(None, sequence_length))
num_samples = tf.shape(inputs)[0]  # useful for later

# word embedding matrix
We = np.random.randn(V, embedding_dim).astype(np.float32)

# weight and bias of the final dense layer (output layer)
Wo = init_weight(hidden_layer_size, K).astype(np.float32)
bo = np.zeros(K).astype(np.float32)

# make them tensorflow variables
tfWe = tf.Variable(We)
tfWo = tf.Variable(Wo)
tfbo = tf.Variable(bo)

# make the rnn unit
rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)

# pass the inputs through the embedding layer to get the output and from that build the cost
x = tf.nn.embedding_lookup(tfWe, inputs)

# TF doesn'tt like 3D objects, so unsttack converts x from a tensor of shape N x T x M
# into a list of length T, where each element is a tensor of shape N x M
x = tf.unstack(x, sequence_length, 1)

# get the rnn output
outputs, states = get_rnn_output(rnn_unit, x, dtype=tf.float32)

# outputs are now of size (T, N, M)
# so make it (N, T, M)
outputs = tf.transpose(outputs, (1, 0, 2))
outputs = tf.reshape(outputs, (sequence_length * num_samples, hidden_layer_size))  # NT x M

# final dense layer
logits = tf.matmul(outputs, tfWo) + tfbo  # we need to flatten our data because matmul only works on 2D tensor
# objects: NT x K
predictions = tf.argmax(logits, 1)
predict_op = tf.reshape(predictions, (num_samples, sequence_length))
labels_flat = tf.reshape(targets, [-1])

# we need to flatten our data because cross entropy cost function  only works on 2D tensor objects and doesn't like
# 3D tensors
cost_op = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels_flat
    )
)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

# init stuff
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# training loop
costs = []
n_batches = len(Ytrain) // batch_size
for i in range(epochs):
    n_total = 0
    n_correct = 0

    t0 = datetime.now()
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    cost = 0

    for j in range(n_batches):
        x = Xtrain[j * batch_size:(j + 1) * batch_size]
        y = Ytrain[j * batch_size:(j + 1) * batch_size]

        # get the cost, predictions, and perform a gradient descent step
        c, p, _ = sess.run(
            (cost_op, predict_op, train_op),
            feed_dict={inputs: x, targets: y})
        cost += c

        # calculate the accuracy
        for yi, pi in zip(y, p):
            # we don't care about the padded entries so ignore them because
            # 0 means padding so we filter out all the enteries where the target is 0

            yii = yi[yi > 0]
            pii = pi[yi > 0]
            n_correct += np.sum(yii == pii)
            n_total += len(yii)

        # print stuff out periodically
        if j % 10 == 0:
            sys.stdout.write(
                "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
                (j, n_batches, float(n_correct) / n_total, cost)
            )
            sys.stdout.flush()

    # get test acc. too
    p = sess.run(predict_op, feed_dict={inputs: Xtest, targets: Ytest})
    n_test_correct = 0
    n_test_total = 0
    for yi, pi in zip(Ytest, p):
        yii = yi[yi > 0]
        pii = pi[yi > 0]
        n_test_correct += np.sum(yii == pii)
        n_test_total += len(yii)
    test_acc = float(n_test_correct) / n_test_total

    print(
        "i:", i, "cost:", "%.4f" % cost,
        "train acc:", "%.4f" % (float(n_correct) / n_total),
        "test acc:", "%.4f" % test_acc,
        "time for epoch:", (datetime.now() - t0)
    )
    costs.append(cost)

plt.plot(costs)
plt.show()
