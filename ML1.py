#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:51:25 2020

@author: imyyounge
"""
# Set up 
import os
import pandas as pd
import matplotlib as plt

os.chdir('/Users/imyyounge/Documents/4_Masters/4_Machine_learning/Nov_2019_Prescribing_Data/Code') 
final = pd.read_csv('final.csv') # Correction: Need to use the dataset outputted by categorical_and_exclusion
def rename_unname(df):
    for col in df:
        if col.startswith('Unnamed'):
            df.drop(col,axis=1, inplace=True)
rename_unname(final)
print(list(final))
final.drop(['left_parent_date'], axis=1, inplace=True) #Drop the columns with NAs so that hopefully things work later

# Separate into training and test data
# Create a column forsize of GP practice - think about funding - small / not  - should be 1 for larger surgeries
def f(row):
    if row['number_of_patients'] < 3000:
        val = 0
    elif row['number_of_patients'] > 2999:
        val = 1
    else:
        val = -1
    return val 
final['binary_#_patients'] = final.apply(f, axis=1)
# Specifying the specific columns
y = final['binary_#_patients'] # Then going to need to make this binary
X = final #Says this is not defined when I try to run the later code
X.drop(['binary_#_patients'], axis=1)
X.drop(['number_of_patients'], axis=1)

# We have set the random seed to be 2, by setting the random_state parameter. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
# This code doesn't work - going to just keep writing code and will come back to debug this afternoon
    # There will be the same proportion of case-controls in the new dataset as original
#Scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train)
    #print(scaler.mean_) # Check it scaled normally

# Don't then need to scale the test data, as the model has been fitted
    #Check this!!!
    
#########################################
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_df = pd.DataFrame(X)
finalnp = final.to_numpy()
X = X.to_numpy()
y = y.to_numpy()
sk = SelectKBest(chi2, k=20)
Xnp_new = sk.fit(X, y)  # Fit the feature selector to the dataset
mask = sk.get_support()
twenty_features = X_df.columns[mask]
twenty_features
sk = SelectKBest(chi2, k=10)
Xnp_ten = sk.fit(X, y) 
mask = sk.get_support()
Xnp_ten = sk.fit(X, y) 
ten_features = X_df.columns[mask] # List the most informative features
ten_features

#Apply the PCA model created earlier to the x_test
X_test2 = pca.transform(X_test) #Need to do the same idea to training data
X_test2_2d = pd.DataFrame(X_test2) # Create a pandas dataframe
X_test2_2d.columns = ['PC1','PC2', 'PC3', 'PC4', 'PC5']
X_test2_2d.head()

#############################################
# KNN
from sklearn import neighbours

knn = neighbors.KNeighborsClassifier() # Name the knn fitter
knn.fit(X_train, y_train) # Fitting the model
y_pred = knn.predict(X_test) # Creating the predictions for the first 100 rows test set
accuracy = np.sum(y_pred == y_test) / len(y_pred) # Comparing to truth to work out accuracy of model
idx_wrong = np.nonzero(y_pred != y_test[:100]) # Creating a group of all the ones that were classified wrong
print('Accuracy = {0:.1f}%.'.format(accuracy * 100)) #Print out

#############################################0
# Linear regression
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression() #Labelling thefeature
linearRegressor.fit(X_train, y_train) # Training the model
y_pred = linearRegressor.predict(X_train) 

plt.scatter(y_train, y_pred)
plt.title('Comparing training data point and the predicted value')
plt.xlabel('Actual y value')
plt.ylabel('Predicted y value')
plt.show()

#Evaluating this model using mean squared error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#Training
print(np.sqrt(mean_squared_error(y_train, y_pred))) #RMSE training set
print(r2_score(y_train, y_pred)) #R2 training
#Test
y_predtest = linearRegressor.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_predtest))) #RMSE test
print(r2_score(y_test, y_predtest)) #R2 test

# Alternative classisification using least squares, from ML practical 1

# Create all possible combinations of attributes. 
# Itertools is a great python library that lets you deal with iterables in efficient ways. 
from itertools import chain, combinations
def all_combinations(attributes):
    """Create all possible combinations when given the attributes"""
    return chain(*map(lambda i: combinations(attributes, i), range(1, len(attributes)+1)))

_attributes = [name for name in column_names if name != 'class']
attribute_combinations = all_combinations(_attributes) #Note that this is an iterable object. 

# Function that takes in a list of attributes, and outputs predictions after carrying out least squares
def return_predictions(attributes, training_data=training_data, testing_data=test_data):    

    X = training_data[attributes].values.reshape(-1, len(attributes))
    _ = np.tile(np.array([1]), [X.shape[0]]).reshape(-1,1)
    X = np.append(_, X, axis=1)
    
    Y = training_data["output"].values.reshape(-1, 1)
    
    X_test = test_data[attributes].values.reshape(-1, len(attributes))
    _ = np.tile(np.array([1]), [X_test.shape[0]]).reshape(-1,1)
    X_test = np.append(_, X_test, axis=1)
    
    # Least squares solution
    W_opt = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))

    predictions = np.matmul(X_test, W_opt)
    
    return predictions

# Function that takes in a predictions vector, and outputs the mean squared error.
def return_mse(predictions, testing_data=test_data):
    Y_test = test_data["output"].values.reshape(-1, 1)
    
    error = Y_test - predictions

    square_error = np.square(error)
    
    mse = np.mean(square_error)
    
    return mse

# evaluate
attribute_combinations = all_combinations(_attributes)
for attributes in attribute_combinations:
    preds = return_predictions(list(attributes))
    print(f"{str(attributes):<70} MSE: {return_mse(preds)}")
    attribute_combinations = all_combinations(_attributes)

for attributes in attribute_combinations:
    preds = return_predictions(list(attributes))
    print(f"{str(attributes):<70} MSE: {return_mse(preds)}")

print(*attribute_combinations)

#GridSearchCV to hypertune parameters
from sklearn.model_selection import GridSearchCV

# The code pattern here is similar to the previous sections. 
# G1) Initiate a GridSearchCV object with the correct model, param_grid, and cv; `cv=k` does a k-fold cross-validation.
grid_search_model = GridSearchCV(DecisionTreeClassifier(random_state=2), {'max_depth':[1, 2, 3, 4, 5, 6]}, cv=15,)

# G2) use the GridSearchCV.fit(X, y) method to run the grid search with cv. 
fitted_grid_search_model = grid_search_model.fit(iris_X, iris_y)

#Accuracy score
full_model_accuracy =  metrics.accuracy_score(y_test, model_2_y_pred)
print(f'Accuracy: {full_model_accuracy}')



#########################################
#CART - need to then play around with the depth of the tree etc 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

    # Gini classifier
dt = DecisionTreeClassifier(max_depth=2, random_state=1, criterion='gini')
dt.fit(X_train, y_train) #Fit dt to the training set
y_pred = dt.predict(X_test) #Predict the test set labels
accuracy_score(y_test, y_pred) #Evaluate tets-set accuracy

    # Information gain classifier
dt = DecisionTreeClassifier(max_depth=2, random_state=1) #Check that defauilt is information gain

    #Decision tree for regression
dt = DecisionTreeRegressor(max_depth=2, random_state=1)
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
print(rmse_dt)

    # Then copy the code over for dealing with bias and variance issues in regression decision trees
    
    # Voting classifier if needed
    
# Plotting the resulting tree 
from sklearn.tree import export_graphviz
from sklearn import metrics
import graphviz 

    # Helper function to plot the decision tree. This uses the graphviz library.
def plot_tree(graph, feature_names=None, class_names=None):
    '''
    This method takes a DecisionTreeClassifier object, along with a list of feature names and target names
    and plots a tree. The feature names and class names can be left empty; they are just there for labelling 
    '''
    dot_data = export_graphviz(graph, out_file=None, 
                      feature_names=feature_names,  
                      class_names=class_names,  
                      filled=True, rounded=True,  
                      special_characters=True) 
    
    graph = graphviz.Source(dot_data)
    
    return graph
plot_tree(fitted_base_model, iris.feature_names, iris.target_names)

#########################################
#  SVM
# Copied straight from ML practical so going to need some adapting
    import numpy as np
import matplotlib.pyplot as plt

# use seaborn plotting defaults
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs

# consider two classes of points which are well separated
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()

# Task 1: Attempt to use linear regression to separate this data using linear regression.
# Note there are several possibilities which separate the data?     
# What happens to the classification of point [0.6, 2.1] (or similar)?

xfit = np.linspace(-1, 3.5) # Create a linear space?
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn') # Scatter plot

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]: 
    plt.plot(xfit, m * xfit + b, '-k')
    
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)    

plt.xlim(-1, 3.5)

plt.show()

# With SVM rather than simply drawing a zero-width line between the 
# classes, we draw a margin of some width around each line, up to the nearest point. 

# Task 2: Draw the margin around the lines you chose in Task 1.

#%%Cell

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
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()

#%% 

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



    






