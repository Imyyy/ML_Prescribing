#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:21:53 2020

@author: imyyounge
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
y = final['binary_#_patients'] # Then going to need to make this binary
X = final #Says this is not defined when I try to run the later code
X.drop(['binary_#_patients'], axis=1, inplace = True)
X.drop(['number_of_patients'], axis=1, inplace = True)
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify = y)
print(list(X_train))
print(list(X_test))
X_train_df = X_train

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  #Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train)    #print(scaler.mean_) # Check it scaled normally

###################################################################################
                            # FITTING AND APPLYING MODEL
###################################################################################
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
logreg = LogisticRegressionCV(random_state= 42, cv=5) 
logreg.fit(X_train, y_train) # Training the model - lets try 5 folds
y_pred = logreg.predict(X_test) 
logreg.get_params()
accuracy = accuracy_score(y_test,y_pred)
logreg.score(X_test, y_pred) # Mean accuracy on the given test data and labels

###################################################################################
                            # OUTPUT METRICS
###################################################################################
#Evaluating this model using mean squared error
# Should be minimising the binary cross entropy here  - need to change things?
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#print('Root mean squared error:')
#print(np.sqrt(mean_squared_error(y_test, y_pred))) #RMSE test
#print('------------------------------')
#print('R squared score:')
#print(r2_score(y_test, y_pred)) #R2 test

# Analysing what is classified
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC curve
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test) # Returns an array with two colums - we choose the second column = p(predicted values =1)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1]) # Fpr = true positive
plt.plot(fpr, tpr, label = 'Logistic Regression ROC Curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
plt.savefig('LogisticRegressionROC.png')

# Area under the ROC curve (AUC)
    # Larger area -> better model
from sklearn.metrics import roc_auc_score
y_pred_prob = logreg.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_prob)
print(auc)












