# importing necessary libraries
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import itertools
import numpy as np 
import csv
import time
import progressbar
 
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
counter = 0
with open('predict.csv', 'r') as f:
    csv_reader = csv.reader(f)
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    for row in csv_reader:
        #Removes the Header from the list and adds it to its own list
        if (isHeader):
            headers.append(row)
            isHeader = False 
            continue
        #Code used for a task, not necessary for general purposes
        if (row[2]=='20' or row[2]=='30'):
            row[2] = '25'

        #Removes target column and puts it into its own list
        labels.append(row[2])
        del row[2]

        encoded = le.fit_transform(row)
        features.append(encoded)
        #Code for progress bar
        counter = counter +1
        bar.update(counter)
progressbar.streams.flush()

# X -> features, y -> label
X = np.asarray(features)
y = np.asarray(labels)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0)

# training a linear SVM classifier
print('Training SVM...')
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

#MLP Neural Network Classifier
print('Training neural network model...')
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)


# creating a report after evaluation
print('SVM')
print(classification_report(y_test, svm_predictions))
print('MLP')
print(classification_report(y_test, mlp_predictions))



#Using the model to make predictions on unlabeled data
'''
print('Predicting on new data...')
with open('test.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        del row[2]
        predict.append(le.fit_transform(row))

predicted = svm_model_linear.predict(predict)
predicted2 = mlp.predict(predict)

total = 0
correct = 0 

for row in predicted:
    if (row == '60'):
        correct = correct + 1
    total = total + 1
    

print('Finished SVM. ' + str(correct) + '/' + str(total) + ' correct predictions')

total = 0
correct = 0 

for row in predicted2:
    if (row == '60'):
        correct = correct + 1
    total = total + 1

print('Finished MLP. ' + str(correct) + '/' + str(total) + ' correct predictions')
'''

#Stopwatch to keep track of how long the algorithms ran for 
stop = time.time()
print('This process took ' + str(stop-start) + ' seconds to complete')

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