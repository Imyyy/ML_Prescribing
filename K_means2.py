#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:16:40 2020

@author: imyyounge
"""
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
final.drop(['left_parent_date'], axis=1, inplace=True) #Drop the columns with NAs so that hopefully things work later
y = final['number_of_patients'] # Then going to need to make this binary
X = final
X.drop(['number_of_patients'], axis=1, inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train) 

#########################################################################
                            # PLOTTING THE DATA
#########################################################################
sns.regplot(x="date_open", y="join_parent_date", data=X_train, fit_reg=False) # PLotting the two largest features

#########################################################################
                                # KMEANS
#########################################################################
from sklearn.cluster import KMeans # create kmeans object
from sklearn.model_selection import GridSearchCV
kmeans = KMeans()
param_grid = {'n_clusters': [1, 2, 3, 4, 5, 6, 7, 8]} # Testing 5, 10 and 20, could also put a range in this section
grid = GridSearchCV(kmeans, param_grid, cv = None)  # cv = None uses the default 5
grid.fit(X_train,y_train) # In-sample acuracy is the accuracy when applied to the training data

kmeans_cv_results = grid.cv_results_ #Store the best performing values in _best_params
kmeans_cv_results
kmeans_best_params = grid.best_params_
kmeans_best_params

# Then output the best features into this model, and fit it -> see the parameters and saving how to plot them
kmeans = KMeans(n_clusters=8) # fit kmeans object to data
kmeans.fit(X_train, y_train)
kmeans_cluster_centres = print(kmeans.cluster_centers_) # print location of clusters learned by kmeans object



# ACCURACY SCORE
y_pred = kmeans.predict(X_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Viewing the output from the k-means clustering

#########################################################################
                            # PLOTTING KMEANS
#########################################################################
# Copied from here: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# Though the following import is not directly being used, it is required
# for 3D projection to work
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
y = X_train
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
    ax.scatter(X['date_open'], X['join_parent_date'], X['items'],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Date opened')
    ax.set_ylabel('Date Joined Organisation')
    ax.set_zlabel('Number of Items')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1


# Plot the ground truth - original code, needs editing
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 0]).astype(np.float)
ax.scatter(X['date_open'], X['nic'], X['items'], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()