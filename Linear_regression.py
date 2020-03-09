#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:22:43 2020

@author: imyyounge
"""
################################################################################
###                              SET UP                                      ###
################################################################################
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
print(list(X_train))


###################################################################################
                                    #REGRESSION
###################################################################################
#Scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # Only need to scale the train, then apply to the test data
scaler.fit(X_train)
scaler.transform(X_train) 
# Applying the Linear regressor
from sklearn.modelselection import LinearRegression
regress = LinearRegression()
regress.fit(X_train, y_train)
print(regress.intercept_)
coeff_df = pd.DataFrame(regress.coef_, X_train.columns, columns=['Coefficient'])
coeff_df #Look at the coefficients from the model

y_pred = regress.predict(X_test) # Make predictions using this model
reg_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(reg_output.head(15))

# Evaluating the regression output
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Plot the outcome of the actual and predicted y values
reg_output['Actual-log'] = np.log(reg_output['Actual'])
reg_output['Predicted-log'] = np.log(reg_output['Predicted'])

sns.regplot(x="Actual", y="Predicted", data=reg_output)
sns.regplot(x="Actual-log", y="Predicted-log", data=reg_output)
############################################# Second copy - is this different?
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
