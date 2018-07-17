# SVM_Multi-class
A template for Multiclass SVM, based off feature rich data sets made of strings and numbers

The template is straightforward to use. First, pip install all of the necessary libraries from the import statements. The next thing to change is the CSV from which you will be getting your data, and the target column. 

The SVM is multi-class, where there exist a finite number of categories but where each member can only belong to a single category with no overlap intended. 

The confusion matrix is printed at the end of the training and validation steps

Commented out is a way to store and load the SVM to disk, so that a larger, complex model can be trained and saved to be use for later predictions

Misc: The code here was tested with approximately 50,000 rows of data containing roughly 60 columns. The program was able to load the data from the CSV, train, and evaluate the machine in only a few minutes time with accuracy in the 90s
