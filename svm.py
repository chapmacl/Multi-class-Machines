# importing necessary libraries
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import itertools
import numpy as np 
import csv
 
features = []
labels = []

print('Loading data...')
#load the data from the csv file into a workable array
with open('predict.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            labels.append(row[2])
            del row[2]
            features.append(row)
            

headers = features[0]
features.pop(0)
headers.insert(2, labels[0])
labels.pop(0)


# X -> features, y -> label
X = features
y = labels
 
testX = []
#The data must be encoded, since SVMs cannot recognize text or string values
le = preprocessing.LabelEncoder()
for row in X:
    testX.append(le.fit_transform(row))


# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(testX, y, random_state = 0)
 
# training a linear SVM classifier
print('Training SVM...')
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
print('SVM Model is: ' + str(accuracy) + ' accurate')
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print(cm)

#For saving the model to Disk
'''
import cPickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(svm_model_linear, fid)    

# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    svm_model_linear_loaded = cPickle.load(fid)
'''