from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
This code explores semantic space analysis using 
VADER (Valence Aware Dictionary for sEntiment Reasoning)
when labels are not available (applied directly to unlabeled text). 

VADER is available in the NLTK and it is sensitive to both 
polarity (+/-ve) and intensity (strength) of emotion.

VADER relies on a dictionary which maps lexical features
to emotion intensities (sentiment scores).

doc. sentiment score = sum(sentiment of its words)
capitalization: adds stronger intensity

"""
#%%
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import config
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#%%

class UnsupervisedSentimentIntensityAnalyzer:
    "returns -ve/neutral/+ve and compound scores of all 3"
    def __init__(self,text_input):
        self.text_input = text_input

    def analyzer(self):
        sid = SentimentIntensityAnalyzer()
        return sid.polarity_scores(self.text_input)

    def review_rating(self):
        scores = self.analyzer()
        if scores['compound'] == 0:
            return f'{self.text_input}: Neutral Review'
        elif scores['compound'] > 0:
            return f'{self.text_input}: Positive Review'
        else:
            return f'{self.text_input}: Negative Review'

    def __str__(self):
        return f'{self.analyzer()}'



#%%

class Unsupervised_Doc_Classifier:
    """
    returns Amazon Review and Movie Review analysis
    """
    def __init__(self, file):
        self.file = file

    def read_df(self):
        df = pd.read_csv(self.file, sep='\t')
        return df

    def print_label_count(self):
        df = self.read_df()
        return f'{df.iloc[:, 0].value_counts()}'

    def detect_remove_NaNs(self):
        df = self.read_df()
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

        #print('Empty strings index: \n', len(blanks), 'blanks: ', blanks)
        df.drop(blanks, inplace=True)
        return df

    def read_amazon_review(self):
        df = self.detect_remove_empty_strings()
        return df

    def score_reviews(self):
        df = self.read_amazon_review()
        sid = SentimentIntensityAnalyzer()
        df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
        return df

    def generate_compound_scores(self):
        df = self.score_reviews()
        df['compound'] = df['scores'].apply(lambda d:d['compound'])
        df['compound_score'] = df['compound'].apply(lambda score:'pos' if score>=0 else 'neg')
        return df

    def compare_compound_score_with_ground_truth(self):
        df = self.generate_compound_scores()
        ground_truth = df['label']
        compound_score_estimated_with_vader = df['compound_score']
        accuracy = accuracy_score(ground_truth,compound_score_estimated_with_vader)
        full_classification_report = classification_report(ground_truth,compound_score_estimated_with_vader)
        confusion_metric_report = confusion_matrix(ground_truth,compound_score_estimated_with_vader)
        return accuracy, full_classification_report, confusion_metric_report

    def __str__(self):
        return f'{self.compare_compound_score_with_ground_truth()}'




#%%

def main():
    text1 = "It is a great day to play tennis"
    a = UnsupervisedSentimentIntensityAnalyzer(text1)
    print(a)

    text2 = "It is the BEST day EVER to play tennis!!"
    a = UnsupervisedSentimentIntensityAnalyzer(text2)
    print(a)

    text3 = "It is the WORST day EVER to play tennis!"
    a = UnsupervisedSentimentIntensityAnalyzer(text3)
    print(a)

    text4 = "The is the BEST MOVIE ever produced!!"
    a = UnsupervisedSentimentIntensityAnalyzer(text4)
    print(a.review_rating())

    file1 = config.AMAZON_REVIEW_DATA
    a = Unsupervised_Doc_Classifier(file1)
    print(a)

    file2 = config.MOVIE_REVIEWS_DATA
    a = Unsupervised_Doc_Classifier(file2)
    print(a)

main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()

