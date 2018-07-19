# importing necessary libraries
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import itertools
import numpy as np 
import csv
import time
 
features = []
labels = []
predict = []
isHeader = True
headers = []

start = time.time()
print('Loading data...')
#load the data from the csv file into a workable array
#The data must be encoded, since SVMs cannot recognize text or string values
le = preprocessing.LabelEncoder()
cv = CountVectorizer()
with open('predict.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        if (isHeader):
            headers.append(row)
            isHeader = False 
            continue
        labels.append(row[2])
        del row[2]
        '''
        encoded = le.fit_transform(row)
        encoded = encoded.reshape(len(encoded), 1)
        oe = OneHotEncoder(sparse=False)
        oe = oe.fit_transform(encoded)
        '''
        features.append(le.fit_transform(row))
            


# X -> features, y -> label
X = features
y = labels

 
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0)
 
# training a linear SVM classifier
print('Training SVM...')
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

#Training a NB classifier
print('Training NB...')
clf = GaussianNB()
clf.fit(X_train, y_train)
nb_predictions = clf.predict(X_test)

print('Training SGDC model...')
sg = linear_model.SGDClassifier()
sg.fit(X_train, y_train)
sg_predictions = sg.predict(X_test)

print('Training decision tree model...')
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
dtc_predictions = dtc.predict(X_test)

print('Training neural network model...')
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)


# creating a confusion matrix
print('SVM')
print(classification_report(y_test, svm_predictions))
print('NB')
print(classification_report(y_test, nb_predictions))
print('SGDC')
print(classification_report(y_test, sg_predictions))
print('DT')
print(classification_report(y_test, dtc_predictions))
print('MLP')
print(classification_report(y_test, mlp_predictions))

stop = time.time()
print('This process took ' + str(stop-start) + ' seconds to complete')

'''
with open('test.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        del row[2]
        predict.append(le.fit_transform(row))

predicted = svm_model_linear.predict(predict)

total = 0
correct = 0 

for row in predicted:
    if (row == '60'):
        correct = correct + 1
    total = total + 1

print('Finished. ' + str(correct) + '/' + str(total) + 'correct predictions')
#For saving the model to Disk
'''

'''
import cPickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(svm_model_linear, fid)    

# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    svm_model_linear_loaded = cPickle.load(fid)
'''