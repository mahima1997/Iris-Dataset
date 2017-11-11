
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:09:53 2017

@author: mahima
"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
#from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data=pd.read_csv('Iris.csv')
scatter_matrix(data)
plt.show()
array=data.values
X = array[:,1:5]
Y = array[:,5]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
#results = []
#names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
	#results.append(cv_results)
	#names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#individual assessing the models without KFold
model = GaussianNB()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_validation)
cnf_matrix = confusion_matrix(Y_predict, Y_validation)
print(cnf_matrix)
print(accuracy_score(Y_predict,Y_validation))
model=LinearDiscriminantAnalysis()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_validation)
cnf_matrix = confusion_matrix(Y_predict, Y_validation)
print(cnf_matrix)
print(accuracy_score(Y_predict,Y_validation))
model=LogisticRegression()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_validation)
cnf_matrix = confusion_matrix(Y_predict, Y_validation)
print(cnf_matrix)
print(accuracy_score(Y_predict,Y_validation))
model=KNeighborsClassifier()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_validation)
cnf_matrix = confusion_matrix(Y_predict, Y_validation)
print(cnf_matrix)
print(accuracy_score(Y_predict,Y_validation))
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_validation)
cnf_matrix = confusion_matrix(Y_predict, Y_validation)
print(cnf_matrix)
print(accuracy_score(Y_predict,Y_validation))
model=SVC()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_validation)
cnf_matrix = confusion_matrix(Y_predict, Y_validation)
print(cnf_matrix)
print(accuracy_score(Y_predict,Y_validation))
