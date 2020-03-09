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










