#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:51:25 2020

@author: imyyounge
"""
# Set up 
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
X.drop(['binary_#_patients'], axis=1, inplace = True)
X.drop(['number_of_patients'], axis=1, inplace = True)

# We have set the random seed to be 2, by setting the random_state parameter. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
X_train.drop(['number_of_patients']) # May already be running, need to check 
print(list(X_train))
X_train_df = X_train
# This code doesn't work - going to just keep writing code and will come back to debug this afternoon
    # There will be the same proportion of case-controls in the new dataset as original
#Scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train) 
    #print(scaler.mean_) # Check it scaled normally

# Don't then need to scale the test data, as the model has been fitted, and the outcome variable here is binary
    #Check this!!!
    
#########################################
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#X_train = X_train.to_numpy()
#y_train = y_train.to_numpy()
sk = SelectKBest(chi2, k=20)
Xnp_new = sk.fit(X_train, y_train)  # Fit the feature selector to the dataset
mask = sk.get_support() # Gets a boolean array
X_train_cols = X_train.columns
twenty_features = X_train.columns[mask]
twenty_features
sk = SelectKBest(chi2, k=10)
Xnp_ten = sk.fit(X_train, y_train) 
mask = sk.get_support()
ten_features = X_train.columns[mask] # List the most informative features
ten_features

#Apply the PCA model created earlier to the x_test
from sklearn.decomposition import PCA
pca = PCA(n_components=20, random_state=42)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum()) # PCA
X_train_pca = pca.transform(X_train) #Need to do the same idea to training data # Create a pandas dataframe
#X_train_pca =  pd.DataFrame(X_train_pca)
#X_train_pca.columns = ['PC1','PC2', 'PC3', 'PC4', 'PC5']
X_train_pca.head()

#### Plots from PCA - see jupyter

#############################################


#############################################0
# Logistic regression - do I need to penalise this?
# Going to try and predict ehether above or below the threshold

from sklearn.linear_model import LogisticRegressionCV #Using CV as it has in
from sklearn.metrics import accuracy_score
logreg = LogisticRegressionCV(cv=5, random_state=0) #Labelling thefeature
logreg.fit(X_train, y_train) # Training the model - lets try 5 folds
y_pred = logreg.predict(X_test) 
logreg.get_params()
accuracy = accuracy_score(y_test,y_pred)
logreg.score(X_test, y_pred) # Mean accuracy on the given test data and labels

plt.scatter(y_test, y_pred)
plt.title('Comparing test data point and the predicted value')
plt.xlabel('Actual y value')
plt.ylabel('Predicted y value')
plt.show()

# Training set info
from matplotlib.colors import ListedColormap
#x_set, y_set = X_train, y_train
x_set = pd.DataFrame.to_numpy(X_train)
y_set = pd.DataFrame.to_numpy(y_train)
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1, step =0.01),
np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1, step =0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x1.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i),label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated')
    plt.show()
    
# Suggested way to plot this from seaborn
tips["big_tip"] = (tips.tip / tips.total_bill) > .175
ax = sns.regplot(x="total_bill", y="big_tip", data=tips, logistic=True, n_boot=500, y_jitter=.03)

# Test set plot
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1, step =0.01),
np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1, step =0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap((‘red’, ‘green’)))
plt.xlim(x1.min(), x1.max())
plt.ylim(x1.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
c = ListedColormap((‘red’, ‘green’))(i),label = j)
plt.title(‘Logistic Regression (Testing set)’)
plt.xlabel(‘Age’)
plt.ylabel(‘Estimated Salary’)
plt.show()

#Evaluating this model using mean squared error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('Root mean squared error:')
print(np.sqrt(mean_squared_error(y_test, y_pred))) #RMSE test
print('------------------------------')
print('R squared score:')
print(r2_score(y_test, y_pred)) #R2 test


#############################################0
# Linear regression - might need to do penalised regression because of the amount of correlation in my data
# Going to try and predict the number of people at each gp surgery,
from sklearn.linear_model import LinearRegression # Want to find the cross validated version
linreg = LinearRegression()#Labelling thefeature
linreg.fit(X_train, y_train) # Training the model
y_pred = linreg.predict(X_test) 

plt.scatter(y_test, y_pred)
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










