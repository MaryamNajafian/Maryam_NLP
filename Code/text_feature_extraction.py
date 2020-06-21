from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
This code contains classes needed for feature extraction
Contains:
    1- Local implementation of TFIDF
    2- Sklearn implementation of TFIDF

tf(t,d)=term freq.: raw count of a number of times a term occurs in a document
idf(t,D) = inv. doc. freq.: log of total #No.of docs/#No. of docs containing the term)
tfidf(t,d,D) = tf(t,d)* idf(t,D)
"""

# %%
import config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics


# %%
class Doc_Classifier:
    """
    Movie +/- review classifier
    Spam message classifier
    """

    def __init__(self, file, text=''):
        self.file = file
        self.text = text

    def read_df(self):
        df = pd.read_csv(self.file, sep='\t')
        # df = df.iloc[:, : 2]
        return df

    def detect_remove_NaNs(self):
        df = self.read_df()
        # print('Missing elements in each column \n', df.isnull().sum())
        df.dropna(inplace=True)
        return df

    def detect_remove_empty_strings(self):
        df = self.detect_remove_NaNs()
        df = df.iloc[:, : 2]
        blanks = []  # start with an empty list

        for i, lb, rv in df.itertuples():  # iterate over the DataFrame
            if type(rv) == str:  # avoid NaN values
                if rv.isspace():  # test 'review' for whitespace
                    blanks.append(i)  # add matching index numbers to the list

        print('Empty strings index: \n', len(blanks), 'blanks: ', blanks)
        df.drop(blanks, inplace=True)
        return df

    def read_messages(self):
        df = self.detect_remove_empty_strings()
        X, y = df['message'], df['label']
        return X, y

    def read_movie_review(self):
        df = self.detect_remove_empty_strings()
        X, y = df['review'], df['label']
        return X, y

    def classify_spam_messages(self):
        X, y = self.read_messages()
        text_clf = self.trained_classify_model(X, y)
        return text_clf

    def classify_movie_reviews(self):
        X, y = self.read_movie_review()
        text_clf = self.trained_classify_model(X, y)
        return text_clf

    def trained_classify_model(self, X, y):
        X, y = X, y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                             ('clf', LinearSVC())])

        # Feed the training data through the pipeline
        text_clf.fit(X_train, y_train)

        # Form a prediction set
        predictions = text_clf.predict(X_test)

        # Report the confusion matrix
        df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam'])
        print(df)
        return text_clf

    def classify_spam_text(self):
        # Test if it is a spam or ham
        text_clf = self.classify_spam_messages()
        print(self.text, text_clf.predict(self.text))

    def classify_movie_review_text(self):
        # Test if it is a + or - review
        text_clf = self.classify_movie_reviews()
        print(self.text, text_clf.predict(self.text))



class Build_Dict_From_Documents():
    def __init__(self, *args):
        self.args = args

    def __str__(self):
        return (str(self.create_vocab_dict()))

    def doc2list(self, file_name):
        with open(file_name) as f:
            return f.read().lower().split()

    def multi_doc2list(self):
        list_all = []
        for doc in self.args:
            list_all.extend(self.doc2list(doc))
        return list_all

    def create_vocab_dict(self):
        vocab = {}
        i = 1
        vocab_list = self.multi_doc2list()
        for word in vocab_list:
            if word in vocab:
                continue
            else:
                vocab[word] = i
                i += 1
        return vocab


# %%
class Feature_Extraction():
    def __init__(self, text_doc, vocab):
        self.text_doc = text_doc
        self.vocab = vocab

    def __str__(self):
        return str(self.vectorizer())

    def vectorizer(self):
        with open(self.text_doc) as f:
            word_vector = [f.name.split('/')[-1]] + [0] * len(self.vocab)
        with open(self.text_doc) as f:
            x = f.read().lower().split()
        for word in x:
            word_vector[self.vocab[word]] += 1
        return word_vector


# %%
def movie_review_classifier(file):
    df = pd.read_csv(file, sep='\t')

    # Detect & remove empty strings
    blanks = []

    for i, lb, rv in df.itertuples():  # iterate over the DataFrame
        if type(rv) == str:  # avoid NaN values
            if rv.isspace():  # test 'review' for whitespace
                blanks.append(i)  # add matching index numbers to the list

    print('Empty strings index: \n', len(blanks), 'blanks: ', blanks)
    df.drop(blanks, inplace=True)

    X, y = df['review'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
def step_by_step_ham_spam_classifier(file):
    df = pd.read_csv(file, sep='\t')
    X_col, y = df['message'], df['label']  # this time we want to look at the text

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(X_train)
    #
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    #  CountVectorizer followed by TfidfTransformer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_col)  # remember to use the original X_train set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # classifier
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam'])
    print(df)


def main():
    # %% Test Build_Dict_From_Documents
    a = Build_Dict_From_Documents(config.TFIDF_DATA_1, config.TFIDF_DATA_2)
    vocab = a.create_vocab_dict()

    # %% Test Feature_Extraction
    b = Feature_Extraction(config.TFIDF_DATA_1, vocab)
    print(b.vectorizer())

    # %% Test movie_review_classifier
    file = config.MOVIE_REVIEWS_DATA_FILE
    movie_review_classifier(file)

    # %% Test Doc_Classifier
    sample_spam_text = ['Congratulations!To get the prize press 55555']
    file = config.SPAM_DATA_FILE

    classifier = Doc_Classifier(file)
    classifier.classify_spam_messages()

    classifier = Doc_Classifier(file, sample_spam_text)
    classifier.classify_spam_text()

    sample_review_text = ['Worst movie ever!']
    file3 = config.MOVIE_REVIEWS_DATA_FILE

    classifier = Doc_Classifier(file3)
    classifier.classify_movie_reviews()

    classifier = Doc_Classifier(file3, sample_review_text)
    classifier.classify_movie_review_text()

#%%
main()
#%%
if __name__ == "__main__":
    print(__doc__)
    main()
