#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:02:09 2020

@author: imyyounge
"""
##################################################################################
                                    #SETUP
##################################################################################
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
#print(list(final))
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

print(list(X_train))
X_train_df = X_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train) 
                                   
##################################################################################
                                    #CART
##################################################################################
#CART - need to then play around with the depth of the tree etc 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

    # Gini classifier
dt_md2 = DecisionTreeClassifier(max_depth=2, random_state=1, criterion='gini')
dt_md2.fit(X_train, y_train) #Fit dt to the training set
y_pred = dt_md2.predict(X_test) #Predict the test set labels
accuracy_score(y_test, y_pred) #Evaluate test - set accuracy

    # Information gain classifier
#dt_info = DecisionTreeClassifier(max_depth=2, random_state=1, criterion='info') #Check that defauilt is information gain
#dt_info.fit(X_train_df, y_train) #Fit dt to the training set
#y_pred = dt_info.predict(X_test) #Predict the test set labels
#accuracy_score(y_test, y_pred)

# Grid searching this
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4, 5]} # Testing 5, 10 and 20, could also put a range in this section
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = None)#takes a dictionary of parameters and values to try out, and a clasfier instance
grid.fit(X_train,y_train) # In-sample acuracy is the accuracy when applied to the training data
grid.cv_results_
grid.best_estimator_
grid.best_params_

# Plot the classificaiton tree with the best parameters
toplotbest = DecisionTreeClassifier(max_depth=10, random_state=1, min_samples_split=2)
toplotbest.fit(X_train, y_train) #Fit dt to the training set
y_pred = toplotbest.predict(X_test) #Predict the test set labels
accuracy_score(y_test, y_pred) 
plot_tree(toplotbest)

    
from sklearn.metrics import roc_curve
y_pred_prob = toplotbest.predict_proba(X_test) # Returns an array with two colums - we choose the second column = p(predicted values =1)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1]) # Fpr = true positive
plt.plot(fpr, tpr, label = 'Logistic Regression ROC Curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('DTC Curve')
plt.show()
plt.savefig('DecisionTreeClassifierROC.png')

from sklearn.metrics import roc_auc_score
y_pred_prob = toplotbest.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_prob)
print(auc)


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

plot_tree(dt_md2)
vc = VotingClassifier(estimators=classifiers)
vc.fit = (x_train, y_train)
y_pred = vc.predict(X_test)
print('Voting classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))

###### ENSEMBLE LEARNING WITH A CART # Where am I going to manage to do this in my code?

##### Then want to how a feature selection chart, for which one is put at the top of the tree most often 
