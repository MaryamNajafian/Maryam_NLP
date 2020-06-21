from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
Unsupervised Topic Modelling methods covered here:
1-Non-negative Matrix Factorization (NMF) 
2-Latent Dirichlet Allocation (LDA)
"""
# %%
import pandas as pd
# imports for lda - slower
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# imports for nmf - faster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import config


# %%
class UnsupervisedTopicSpotting:
    """Apply LDA /NMF to Select Top N Topics And Corresponding TopK Words Per Row"""

    def __init__(self, topic_spotting_method, k_words, n_topics, max_doc_freq, min_doc_freq, rand_state, doc_name,
                 col_name):
        self.k_words = k_words
        self.n_topics = n_topics
        self.doc_name = doc_name
        self.col_name = col_name
        self.max_doc_freq = max_doc_freq
        self.min_doc_freq = min_doc_freq
        self.rand_state = rand_state
        self.topic_spotting_method = topic_spotting_method  # Options ['lda' |'nmf' ]

    def read_doc(self):
        all_articles = pd.read_csv(self.doc_name)
        return all_articles

    def extract_doc_term_matrix(self):
        """
        output doc term matrix dimensions: No. Articles x No. Words
        for the word to show up in the count vectorizer it should belong to the min/max doc.freq. criteria
        min/max doc.freq.: you can pass in words that show up in x many docs or x% of docs
        """
        all_articles = self.read_doc()

        if self.topic_spotting_method == 'lda':
            count_vectorizer = CountVectorizer(max_df=self.max_doc_freq, min_df=self.min_doc_freq, stop_words='english')
            doc_term_matrix = count_vectorizer.fit_transform(all_articles[self.col_name])
            return doc_term_matrix, count_vectorizer

        elif self.topic_spotting_method == 'nmf':
            tfidf_vectorizer = TfidfVectorizer(max_df=self.max_doc_freq, min_df=self.min_doc_freq, stop_words='english')
            doc_term_matrix = tfidf_vectorizer.fit_transform(all_articles[self.col_name])
            return doc_term_matrix, tfidf_vectorizer

        else:
            return None

    def train_topic_spotter(self):
        doc_term_matrix, vectorizer = self.extract_doc_term_matrix()

        if self.topic_spotting_method == 'lda':
            lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=self.rand_state)
            lda.fit(doc_term_matrix)
            return lda

        elif self.topic_spotting_method == 'nmf':
            nmf = NMF(n_components=self.n_topics, random_state=self.rand_state)
            nmf.fit(doc_term_matrix)
            return nmf

    def words_in_topic_spotter_list_of_vocab(self):
        doc_term_matrix, vectorizer = self.extract_doc_term_matrix()
        return vectorizer.get_feature_names()

    def count_num_words_in_topic_spotter_list_of_vocab(self):
        return len(self.words_in_topic_spotter_list_of_vocab())

    def return_shape_of_topic_spotter_components(self):
        topic_spotter = self.train_topic_spotter()
        return topic_spotter.components_.shape

    def extract_k_most_probable_words_for_topic_t(self, topic_t):
        top_k_words = self.k_words * -1
        topic_spotter = self.train_topic_spotter()
        single_topic = topic_spotter.components_[topic_t]
        single_topic.argsort()  # Returns the indices that would sort this array.
        list_of_words = single_topic.argsort()[top_k_words:]  # Top 10 words for this topic
        return list_of_words

    def extract_k_most_probable_words_for_all_n_topics(self):

        """
            for nmf : K words highest coefficient
            for lda : K words with highest probability
        """
        top_k_words = self.k_words * -1
        doc_term_matrix, vectorizer = self.extract_doc_term_matrix()
        topic_spotter = self.train_topic_spotter()
        for index, topic in enumerate(topic_spotter.components_):
            top_k_words_per_topic = [vectorizer.get_feature_names()[i] for i in topic.argsort()[top_k_words:]]
            print(f"The top {self.k_words} words for topic #{index} is: {top_k_words_per_topic}")

    def topic_prob_distribution_per_article_row_in_dataframe(self):
        doc_term_matrix, vectorizer = self.extract_doc_term_matrix()
        topic_spotter = self.train_topic_spotter()
        prob_of_each_row_belonging_to_t_topics = topic_spotter.transform(doc_term_matrix)
        return prob_of_each_row_belonging_to_t_topics

    def return_prob_distribution_of_article_x_belonging_to_t_topics(self, x):
        article_x = x
        prob_of_each_row_belonging_to_t_topics = self.topic_prob_distribution_per_article_row_in_dataframe()
        return prob_of_each_row_belonging_to_t_topics[article_x].round(2)

    def return_most_probable_topic_of_article_x(self, x):
        article_x = x
        prob_list = self.return_prob_distribution_of_article_x_belonging_to_t_topics(article_x)
        most_probable_topic = prob_list.argmax()
        return most_probable_topic

    def return_new_df_with_accompanied_most_probable_topic(self):
        df = self.read_doc()
        topic_results = self.topic_prob_distribution_per_article_row_in_dataframe()
        df['Topic'] = topic_results.argmax(axis=1)
        return df

    def return_final_df_with_topic_name_assigned_to_each_topic(self, mytopic_list):
        df = self.return_new_df_with_accompanied_most_probable_topic()
        mytopic_dict = {i + 1: j for i, j in enumerate(mytopic_list)}
        df['Topic Label'] = df['Topic'].map(mytopic_dict)
        return df


# %%
def main():
    # Select an option
    option = 3

    # Options: required
    doc_col_options = [(config.EXAMPLE_ARTICLES, 'Article'), (config.EXAMPLE_QUESTIONS, 'Question')]
    topic_spotter_options = ['lda', 'nmf']
    k_words, n_topics, max_doc_freq, min_doc_freq, rand_state = 10, 7, 0.85, 2, 42
    # optional: after reading the words associated with each topic we can add them to the df
    mytopic_list = []
    # E.g. for option 2: mytopic_list = ['health', 'election', 'legislation', 'international politics', 'domistic politics', 'music','education']


    # Set values
    if option == 1:
        i, j = 0, 0
        doc_name, col_name, topic_spotting_method = doc_col_options[i][0], doc_col_options[i][1], topic_spotter_options[
            j]
    elif option == 2:
        i, j = 0, 1
        doc_name, col_name, topic_spotting_method = doc_col_options[i][0], doc_col_options[i][1], topic_spotter_options[
            j]
    elif option == 3:
        i, j = 1, 1
        doc_name, col_name, topic_spotting_method = doc_col_options[i][0], doc_col_options[i][1], topic_spotter_options[
            j]

    else:
        print("The default option 1 is selected.\n")
        option = 1

    a = UnsupervisedTopicSpotting(topic_spotting_method, k_words, n_topics, max_doc_freq, min_doc_freq,
                                  rand_state, doc_name, col_name)
    # a.extract_k_most_probable_words_for_all_n_topics()
    # a.return_prob_distribution_of_article_x_belonging_to_t_topics(1)
    b = a.return_new_df_with_accompanied_most_probable_topic()
    print(b.head())

    if len(mytopic_list) >= 1:
        bb = a.return_final_df_with_topic_name_assigned_to_each_topic(mytopic_list)
        bb.head()

#%%
main()
#%%
if __name__ == "__main__":
    print(__doc__)
    main()

"""
LDA is a graphical model used for topic discovery.
It represents docs. as mixture of topics comprising words
with certain probabilities. 

Assumptions of LDA for topic modeling
    * documents with similar topics use similar groups of 
    words (documents are prob. dist. over latent topics)
    * latent topics can be found by searching for groups 
    of words that frequently occur together in documents 
    across corpus (topics are prob. dist. over words)
    
User must:
    * decide on the # of K topics in the document (set K topics to discover)
    * interpret what the topics are from word prob. distribution per topic
    
NMF performs an iterative EM optimization 
for estimating (W and H) where k is the number of topics: 
A(data matrix: nxm) = W(basis vectors: nxk) x H(coefficient matrix: kxm) 

    * Call tf-idf-vectorizer tp
        1.1- construct a doc. term matrix, 
        1.2- weight normalize (apply TF-IDF to A), 
        1.3- unit length normalize 
    * initialize factors for A with NNDSVD (non-negative double singular val. decomp.) 
    * apply projected gradient NMF to A

It performs dimensionality reduction and clustering
It can be used with TF-IDF to model topics across documents

"""
