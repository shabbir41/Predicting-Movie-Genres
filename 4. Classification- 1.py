# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:44:11 2019

@author: Shabbir Khan
"""

from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost, numpy
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support as score

data_df = pd.read_csv("movie_genre_new.csv", delimiter='\t')
data_x = data_df[['plots']].as_matrix()
data_y = data_df[['genres']].as_matrix()
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x,data_y,test_size=0.2, random_state=42)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(y_train)
test_y = encoder.fit_transform(y_test)
train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

xtrain_tfidf =  tfidf_vect.fit_transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)
xtrain_tfidf.shape
xtest_tfidf.shape

def train_model(classifier , feature_vector_train, label , feature_vector_valid):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    ac = metrics.accuracy_score(predictions ,test_y)
    jc = metrics.jaccard_similarity_score(predictions, test_y)
    f = metrics.f1_score(test_y, predictions, average="weighted")
    return [ac,jc,f]

accuracy = list()
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf , train_y , xtest_tfidf)
print("NB, TFIDF Vectors: ")
print("Accuracy Score:",accuracy[0])
print("Jaccard Similarity Score", accuracy[1])
print("Fscore", accuracy[2])
accuracy2 = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xtest_tfidf.tocsc())
print("Xgb, Count Vectors: ", accuracy2)