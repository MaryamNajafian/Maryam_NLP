"""
This file contains the path to access all the data consumed by different classes
"""

# current directory
# CURRENT_DIR = 'Path/to/dir/'
CURRENT_DIR = '/Users/mn1048144/Desktop/NLP/'

# data
DATA_DIR = CURRENT_DIR + 'Data/'

# Data for pdf_doc_to_text_read_write.py
PDF_DATA_FILE = DATA_DIR + 'US_Declaration.pdf'
PDF_DATA_PAGE = DATA_DIR + 'MY_NEW_FILE.pdf'
TEXT_DATA_FILE = DATA_DIR + 'test.txt'

# Data for text_classification.py
SPAM_DATA_FILE = DATA_DIR + 'smsspamcollection.tsv'
MOVIE_REVIEWS_DATA_FILE = DATA_DIR + 'moviereviews.tsv'

# Data for text_feature_extraction.py
TFIDF_DATA_1 = DATA_DIR + '1.txt'
TFIDF_DATA_2 = DATA_DIR + '2.txt'

# Data for unsupervised_sentiment_classification
AMAZON_REVIEW_DATA = DATA_DIR + 'amazonreviews.tsv'
MOVIE_REVIEWS_DATA = DATA_DIR + 'moviereviews.tsv'

# Data for unsupervised_topic_spotting.py
EXAMPLE_ARTICLES = DATA_DIR + 'npr.csv' #11992 articles
EXAMPLE_QUESTIONS = DATA_DIR + 'quora_questions.csv'

# Data for text_generation.py
EXAMPLE_BOOK_CHAPTER = DATA_DIR + 'moby_dick_four_chapters.txt'

# Data for chatbot.py
CHATBOT_QA_TRAIN = DATA_DIR + 'train_qa.txt'
CHATBOT_QA_TEST = DATA_DIR + 'test_qa.txt'

# Pre-trained wordvectors
"""
* Access pre-trained GloVe vectors: `https://nlp.stanford.edu/projects/glove/` Direct link: `http://nlp.stanford.edu/data/glove.6B.zip`
* Access pre-trained Word2Vec vectors:  (warning: takes quite awhile) `https://code.google.com/archive/p/word2vec/`
* Access pre-trained Word2Vec vectors (direct link): `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing`
* Access Reuters train and test sets: `https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-train-all-terms.txt` `https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-test-all-terms.txt`
* Access  Brown corpora, Run the Python interpreter and type the commands:
    >>> import nltk
    >>> nltk.download()
    Then from the menu, select Corpora tab > select Brown corpus > install

"""
LARGEFILE_DIR = CURRENT_DIR + ' /'
GLOVE_PRETRAINED = LARGEFILE_DIR + 'glove.6B/glove.6B.50d.txt'
W2V_PRETRAINED = LARGEFILE_DIR + 'GoogleNews-vectors-negative300.bin'
REUTERS_TRAIN = LARGEFILE_DIR + 'r8-train-all-terms.txt'
REUTERS_TEST = LARGEFILE_DIR + 'r8-test-all-terms.txt'
WIKIPEDIA_DATA = LARGEFILE_DIR + 'enwiki-preprocessed'
TEXT8_DATA = LARGEFILE_DIR + 'text8'
CHUNKING_DATA = LARGEFILE_DIR + 'chunking'
TWITTER_NER_DATA = DATA_DIR + 'twitter_ner.txt'
STANFORD_MOVIE_REVIEW = DATA_DIR + 'trainDevTestTrees_PTB/'
# POS tagging

# NER