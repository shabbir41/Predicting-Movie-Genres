# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:04:32 2019

@author: Shabbir Khan
"""

from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost, numpy
import pandas as pd
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt


data_df = pd.read_csv("movie_genre_new.csv", delimiter='\t')
data_x = data_df[['plots']].as_matrix()
data_y = data_df[['genres']].as_matrix()
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x,data_y,test_size=0.2, random_state=42)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(y_train)
test_y = encoder.fit_transform(y_test)
train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
xtrain_tfidf_ngram =  tfidf_vect_ngram.fit_transform(train_x)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)
predictioned = test_y
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def train_model(classifier , feature_vector_train, label , feature_vector_valid):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    ac = metrics.accuracy_score(predictions ,test_y)
    jc = metrics.jaccard_similarity_score(predictions, test_y)
    f = metrics.f1_score(test_y, predictions, average="weighted")
    actu = pd.Series(feature_vector_valid,name='Actual')
    pred = pd.Series(predictions, name='Predicted')
    df_conf = pd.crosstab(actu,pred)
    plot_confusion_matrix(df_conf)
    return [ac,jc,f]

'''accuracy3 = list()'''
accuracy3 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy3)
print("Hit Ratio", accuracy3)
