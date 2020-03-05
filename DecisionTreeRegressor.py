#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:16:59 2020

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
X_train.drop(['number_of_patients']) # May already be running, need to check 
print(list(X_train))
X_train_df = X_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train) 

###################################################################################
                            # DECISION TREE REGRESSOR
###################################################################################
# Look back at DataCamp Notes
    # Don't need to scale the data before using a CART
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
dt = DecisionTreeRegressor(max_depth=5, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
print(rmse_dt)

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

plot_tree(dt)


    # Then copy the code over for dealing with bias and variance issues in regression decision trees
