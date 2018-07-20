# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 21:47:13 2018

@author: aaryan
"""

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

# --Loading The DataSet--
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion', 
         'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitosis', 'class']
df = pd.read_csv(url, names=names)

# --Preprocessing The Data--
df.replace('?', -99999, inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)

# --Print The Shape--
print(df.shape)

# --DataSet Visualiztion--
print(df.loc[0])
print(df.describe())

# --Plot Histograms For Each Variable--
df.hist(figsize = (10,10))
plt.show()

# --Scatter Plot To See RelationShip B/W There Variables--
scatter_matrix(df, figsize = (18,18))
plt.show() # Result Shows Linear Relations That We Could Use, Therefor LinearRegression Not A Good Choice!!!
 
# --Creating X & Y DataSets For Training--
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# --Specifying Testing Options--
seed = 8
scoring = 'accuracy'

# --Defining The Model To Train--
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC()))

# --Evaluating Each Model In Turn--
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"% (name, cv_results.mean(), cv_results.std())
    print(msg)

# --Make Predictions On VAlidation DataSet--
# Accuracy - ratio of correctly predicted observation to the total observations. 
# Precision - (false positives) ratio of correctly predicted positive observations to the total predicted positive observations
# Recall (Sensitivity) - (false negatives) ratio of correctly predicted positive observations to the all observations in actual class - yes.
# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false 
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

clf = SVC()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,8]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction) # Predicts that the cell is cancerous



