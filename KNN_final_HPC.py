#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:02:07 2020

@author: imyyounge
"""
################################### SETUP ###############################
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
os.chdir('/Users/imyyounge/Documents/4_Masters/4_Machine_learning/Nov_2019_Prescribing_Data/Code') 
final = pd.read_csv('final.csv') # Correction: Need to use the dataset outputted by categorical_and_exclusion
def rename_unname(df):
    for col in df:
        if col.startswith('Unnamed'):
            df.drop(col,axis=1, inplace=True)
rename_unname(final)
print(list(final))
final.drop(['left_parent_date'], axis=1, inplace=True)
def f(row):
    if row['number_of_patients'] < 3000:
        val = 0
    elif row['number_of_patients'] > 2999:
        val = 1
    else:
        val = -1
    return val 
final['binary_#_patients'] = final.apply(f, axis=1)
y = final['binary_#_patients'] # Then going to need to make this binary
X = final #Says this is not defined when I try to run the later code
X.drop(['binary_#_patients'], axis=1, inplace = True)
X.drop(['number_of_patients'], axis=1, inplace = True)
from sklearn.model_selection import train_test_split # TRAINING AND TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
print(list(X_train))
X_train_df = X_train
from sklearn.preprocessing import StandardScaler # SCALING THE DATA
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train) 

###################################################################################
                            # KNN ROUND 1
###################################################################################
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier() # Name the knn fitter
param_grid = {'n_clusters': [1, 2, 3, 4, 5, 6, 7, 8]} # Testing 5, 10 and 20, could also put a range in this section
grid = GridSearchCV(knn, param_grid, cv = None)  # cv = None uses the default 5
grid.fit(X_train,y_train) # Fitting the model
y_pred = knn.predict(X_test) # Creating the predictions for the first 100 rows test set

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 # In-sample acuracy is the accuracy when applied to the training data

kmeans_cv_results = grid.cv_results_ #Store the best performing values in _best_params
kmeans_cv_results
kmeans_best_params = grid.best_params_
kmeans_best_params

# Then output the best features into this model, and fit it -> see the parameters and saving how to plot them
kmeans = KMeans(n_clusters=8) # fit kmeans object to data
kmeans.fit(X_train, y_train)
kmeans_cluster_centres = print(kmeans.cluster_centers_)

# Trying to plot the number of k values against the amount of errror for that value


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# Need to make a meshgrid to flow through the areas


# There is an arror here somewhere, need to fix it at some point
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Loading some example data
X = X_train
y = y_train

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)


###################################################################################
                            # PLOTTING DECISION REGIONS
###################################################################################
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = y.min() - 1, y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()

###################################################################################
                            # PLOTTING ROUND 2
###################################################################################

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Edited version of code above
X = X_train
y = y_train
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_6', KMeans(n_clusters=6)),
              ('k_means_iris_4', KMeans(n_clusters=4)),
              ('k_means_iris_2', KMeans(n_clusters=2))]

fignum = 1
titles = ['8 clusters', '6 clusters', '4 clusters', '2 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 4))
    ax = Axes3D(fig, rect=[0, 0, .95, 1])
    est.fit(X)
    labels = est.labels_

    ax.scatter(X['date_open'], X['ccg_code1'], X['items'],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Date opened')
    ax.set_ylabel('Date joined parent organisation')
    ax.set_zlabel('Items')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1


######################## Unsupervised KNN #######################
from sklearn.neighbors import NearestNeighbors