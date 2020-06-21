from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
This code contains examples of
visualizations and modeling
"""
# %%
def main():

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn import metrics

    import config
    #%%
    df = pd.read_csv(config.SPAM_DATA_FILE, sep = '\t')
    df.head()
    #%% column stats
    general_statistics_on_columns = df.describe()
    general_statistics_on_specific_column = df['punct'].describe()
    check_missing_values = df.isnull()
    column_wise_totall_null_count = df.isnull().sum()
    row_count = len(df)
    column_unique_vals = df['label'].unique()
    count_unique_val_counts = df['label'].value_counts()

    #%% column selection
    X = df[['length','punct']]
    X = df.drop(['label','message'], axis=1)

    #%% Create Feature and Label sets > train test split> model training
    df = pd.read_csv(config.SPAM_DATA_FILE, sep = '\t')

    X = df[['length','punct']]  # note the double set of brackets
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('Training Data Shape:', X_train.shape)
    print('Testing Data Shape: ', X_test.shape)

    lr_model = LogisticRegression(solver='lbfgs') #https://en.wikipedia.org/wiki/Limited-memory_BFGS
    lr_model.fit(X_train, y_train)

    predictions = lr_model.predict(X_test) # Create a prediction set:
    print(metrics.confusion_matrix(y_test,predictions)) # Print a confusion matrix
    # You can make the confusion matrix less confusing by adding labels:
    df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
    print(df)
    #%%
    # Print a classification report
    print(metrics.classification_report(y_test,predictions))
    # Print the overall accuracy
    print(metrics.accuracy_score(y_test,predictions))
    #%%
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    predictions = nb_model.predict(X_test)
    print(metrics.confusion_matrix(y_test,predictions))
    print(metrics.classification_report(y_test,predictions))
    print(metrics.accuracy_score(y_test,predictions))
    #%%
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    svc_model = SVC(gamma='auto')
    svc_model.fit(X_train,y_train)
    predictions = svc_model.predict(X_test)
    print(metrics.confusion_matrix(y_test,predictions))
    print(metrics.classification_report(y_test,predictions))
    print(metrics.accuracy_score(y_test,predictions))

    #%% visualization
    df = pd.read_csv(config.SPAM_DATA_FILE, sep = '\t')
    plt.xscale('log')
    bins = 1.15**(np.arange(0,50))
    plt.hist(df[df['label'] == 'ham']['length'],bins=bins,alpha=0.8)
    plt.hist(df[df['label'] == 'spam']['length'],bins=bins,alpha=0.8)
    plt.legend(('ham','spam'))
    plt.show()
    #%% visualization
    plt.xscale('log')
    bins = 1.5**(np.arange(0,15))
    plt.hist(df[df['label']=='ham']['punct'],bins=bins,alpha=0.8)
    plt.hist(df[df['label']=='spam']['punct'],bins=bins,alpha=0.8)
    plt.legend(('ham','spam'))
    plt.show()

#%%
main()
#%%
if __name__ == "__main__":
    print(__doc__)
    main()
