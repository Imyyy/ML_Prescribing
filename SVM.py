#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:20:14 2020

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
X_train.drop(['number_of_patients']) # May already be running, need to check 
print(list(X_train))
X_train_df = X_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train) 


##################################################################################
                                    #SVM
##################################################################################
# Copied straight from ML practical so going to need some adapting
# With SVM rather than simply drawing a zero-width line between the classes, we draw a margin of some width around each line, up to the nearest point. 

# Task 2: Draw the margin around the lines you chose in Task 1.

# For SVM the line that maximises the margin is the optimal model

# Task 3: Use the sklearn package to build a support vector classifier using a linear kernel
# (hint: you will need from sklearn.svm import SVC). Plot the decision fuction on the data

from sklearn.svm import SVC # "Support vector classifier"

def plot_svc_decision_function(model, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    X, Y = np.meshgrid(x,y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    

    ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

model = SVC(kernel='linear', C=1E10, gamma = 0.1)
model.fit(X_train, y_train) # Takes a while to fit

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()

# Task 4: Change the number of points in the dataset using X = X[:N] and y = y[:N]
# and build the classifier again using a linear kernel
# Plot the decision function. Do you see any differences?

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.50)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10, gamma = 0.1)
    model.fit(X, y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))
plt.show()
    
## So far we have considered linear boundaries but this is not always the case

## Consider the new dataset
    
from sklearn.datasets.samples_generator import make_circles
X2, y2 = make_circles(100, factor=.1, noise=.1)

#Task 5: Build a classifier using a linear kernel and plot the decision making function

clf = SVC(kernel='linear', gamma = 0.1).fit(X2, y2)

plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.show()

# These results should look wrong so we will try something else

# Consider projecting our data into a 3D plane
r = np.exp(-(X2 ** 2).sum(1))

from mpl_toolkits import mplot3d

ax = plt.subplot(projection='3d')
ax.scatter3D(X2[:, 0], X2[:, 1], r, c=y2, s=50, cmap='autumn')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')

plt.show()

# Looking at the data it is now clear to see that we could draw a linear plane through
# it in the 3D space and classify the data. We can then project back to the 2D
# space. This is what the 'rbf' kernel does.

#Task 6: Try building a classifier using the 'rbf' kernel
clf = SVC(kernel='rbf', C=1E6, gamma = 0.1)
clf.fit(X2, y2)


plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none')
plt.show()


# Task 7: Go back to your original dataset (ie. make blobs) and try using different kernels 
# to build the classifier and plot the results
# Compare the differences between the models

for ker in ['linear', 'poly', 'rbf']:
    model = SVC(kernel = ker, C=1E10, gamma = 0.1)
    model.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model)
    plt.show()

## So far we have looked at clearly delineated data. Consider the following dataset
## where the margins are less clear

X3, y3 = make_circles(n_samples=100, factor=0.2, noise = 0.35)
plt.scatter(X3[:, 0], X3[:, 1], c=y3, s=50, cmap='autumn')
plt.show()

## SVM has a tuning parameter C which softerns the margins. For very large C, 
## the margin is hard, and points cannot lie in it. For smaller $C$, the margin 
# is softer, and can grow to encompass some points.

# Task 8: Try experimenting with different values of C and see what different
# results you get

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='rbf', C=C, gamma = 0.1).fit(X3, y3)
    axi.scatter(X3[:, 0], X3[:, 1], c=y3, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.set_title('C = {0:.1f}'.format(C), size=14)

plt.show()    
    
# Task 9: Use GridSearchCV (hint: from sklearn.model_selection import GridSearchCV)
# to find the optimum parameters for C. 

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 5, 10, 100], 'gamma': [0.01, 0.1, 0.3, 0.5]}

model = SVC(kernel='rbf', C=1, gamma = 0.1).fit(X3, y3)

grid = GridSearchCV(model, param_grid)

grid.fit(X3, y3)
print(grid.best_params_)

model = grid.best_estimator_


plt.scatter(X3[:, 0], X3[:, 1], c=y3, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()