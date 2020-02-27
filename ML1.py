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
toycomp = pd.read_csv('Combined_TOYCOMP_BNF_NHS_data.csv') # Using the BNF version of the dataframe
def rename_unname(df):
    for col in df:
        if col.startswith('Unnamed'):
            df.drop(col,axis=1, inplace=True)
rename_unname(toycomp)
print(list(toycomp.columns))

# Separate into training and test data
y = toycomp['number_of_patients']
X = toycomp
X.drop('number_of_patients', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test y_train, y_test = train_test_split(X, y, test_size = 0.2) 
# There will be the same proportion of case-controls in the new dataset as original


# Pre prepping data - some of this has already been done in previous scripts
#String columns should already be numeric by now

#GridSearchCV to hypertune parameters

#Accuracy score


# Feature selection
from sklearn.feature_selection import chi2, SelectKBest
bestselectedcolumns = SelectKBest(chi2, k=20)
which_selected = sk.fit(X, y).get_support() #Returns the index of the selected columns
X.columns[which_selected]









