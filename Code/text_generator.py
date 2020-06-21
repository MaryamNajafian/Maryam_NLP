from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
Text generation
1- process, clean, tokenize text
2- create sequences
"""

#%%
import spacy
import numpy as np
import random
from random import randint
from pickle import dump, load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from keras.utils import to_categorical

import config


# %%

class TextGenerator:
    """
    Text generation with neural networks
    In this example we feed the model trained on 4 chapters of a book
    with a seed example sentence of length 25
        * if its too short it will be 0 padded
        * if its too long the beginning will be truncated

    """

    def __init__(self, doc_example, seed_text='', num_gen_words=20, train_seq_len=25, rand_seed=101):
        self.doc_example = doc_example
        self.train_len = train_seq_len + 1
        self.num_gen_words = num_gen_words
        self.rand_seed = rand_seed
        if seed_text == '':
            self.seed_text = self.generate_random_seed_text()
        else:
            self.seed_text = seed_text

    def __str__(self):
        generated_string = self.generate_new_text()
        return f"generated string: {generated_string}"

    def read_file(self):
        with open(self.doc_example) as f:
            str_text = f.read()
        return str_text

    def return_punction_string(self):
        all = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
        return all

    def tokenize_and_clean_text(self):
        nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        nlp.max_length = 1198623  # first four chapters
        doc_text = self.read_file()
        punctuation_string = self.return_punction_string()
        punctuation_removed_text = [token.text.lower() for token in nlp(doc_text) if
                                    token.text not in punctuation_string]
        number_of_tokens = len(punctuation_removed_text)  # 11394
        return punctuation_removed_text, number_of_tokens

    def split_text_to_sequences_of_length_n(self):
        """create n text sequences with length 26 words"""
        punctuation_removed_text, number_of_tokens = self.tokenize_and_clean_text()
        list_of_text_sequences = []
        for i in range(self.train_len, number_of_tokens):
            seq = punctuation_removed_text[i - self.train_len:i]
            list_of_text_sequences.append(seq)
        return list_of_text_sequences

    def text_sequence_tokenizer(self):
        """pass 25 words in a sentence --> nextwork predicts 26th word"""
        list_of_text_sequences = self.split_text_to_sequences_of_length_n()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list_of_text_sequences)  # [90,7,98,96,8,3...]
        token_sequences = tokenizer.texts_to_sequences(list_of_text_sequences)
        return tokenizer, token_sequences

    def vocab_size_identifier(self):
        tokenizer, token_sequences = self.text_sequence_tokenizer()
        vocab_size = len(tokenizer.word_counts)  # 2709
        return vocab_size

    def print_id_to_word_to_wordcount_for_nth_sequence(self, n):
        tokenizer, token_sequences = self.text_sequence_tokenizer()
        for i in token_sequences[n]:
            print(f"index {i}: {tokenizer.index_word[i]}, word_count: {tokenizer.word_counts[tokenizer.index_word[i]]}")

    def sequence_to_data_and_labels(self):
        vocab_size = self.vocab_size_identifier()
        tokenizer, token_sequences = self.text_sequence_tokenizer()
        np_sequences = np.array(token_sequences)
        X = np_sequences[:, :-1]  # for all columns except last column (n columns) grab all rows
        y = np_sequences[:, -1]  # grab  the word in last column ((n+1)th column) all rows
        y = to_categorical(y, num_classes=vocab_size + 1)
        return X, y

    def sequence_len(self):
        X, y = self.sequence_to_data_and_labels()
        seq_len = X.shape[1]
        return seq_len

    def create_model(self):
        """
        Add Embedding: positive ints -> dense vectors of fixed size
        Add 2 X LSTM units
        Add 2 x Dense layers: 'relu' and 'softmax'
        compile model with 'categorical_crossentropy' loss so each vocab is its own category
        """
        vocab_size = self.vocab_size_identifier() + 1
        seq_len = self.sequence_len()
        no_neurons = seq_len * 6

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=seq_len, input_length=seq_len))
        model.add(LSTM(units=no_neurons, return_sequences=True))
        model.add(LSTM(units=no_neurons))
        model.add(Dense(units=no_neurons, activation='relu'))
        model.add(Dense(units=vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def save_trained_model_and_tokenizer(self):
        """
        saves the model and tokenizer
        batch_size = Number of sequences you want to pass at a time
        """
        model = self.create_model()
        X, y = self.sequence_to_data_and_labels()
        tokenizer, token_sequences = self.text_sequence_tokenizer()
        # fit model
        model.fit(X, y, batch_size=128, epochs=300, verbose=1)
        # save the model to file
        model.save('my_model.h5')
        # save the tokenizer
        dump(tokenizer, open('my_tokenizer', 'wb'))

        return model, tokenizer

    def generate_random_seed_text(self):
        list_of_text_sequences = self.split_text_to_sequences_of_length_n()
        random.seed(self.rand_seed)
        random_pick = random.randint(0, len(list_of_text_sequences))
        random_seed_text = list_of_text_sequences[random_pick]
        seed_text = ' '.join(random_seed_text)
        return seed_text

    def generate_new_text(self):
        """
        model : model that was trained on text data
        tokenizer : tokenizer that was fit on text data
        seq_len : length of training sequence
        seed_text : raw string text to serve as the seed (length 25 here)
        num_gen_words : number of words to be generated by model
        """

        output_text = []
        input_text = self.seed_text  # Initial Seed Sequence
        seq_len = self.sequence_len()
        model, tokenizer = self.save_trained_model_and_tokenizer()
        num_gen_words = self.num_gen_words

        # Create num_gen_words
        for i in range(num_gen_words):
            # Take the input text string and encode it to a sequence
            encoded_text = tokenizer.texts_to_sequences([input_text])[0]
            # Pad sequences to our trained rate (e.g. 25 words)
            pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
            # Predict Class Probabilities for each word in vocabulary
            pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
            # Grab word
            pred_word = tokenizer.index_word[pred_word_ind]
            # Update the sequence of input text (shifting one over with the new word)
            input_text += ' ' + pred_word
            output_text.append(pred_word)

        # Make it look like a sentence.
        return ' '.join(output_text)


#%%
def main():
    doc_example = config.EXAMPLE_BOOK_CHAPTER
    a = TextGenerator(doc_example)
    print(a)


#%%
main()
#%%
if __name__ == "__main__":
    print(__doc__)
    main()
