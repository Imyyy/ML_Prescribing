#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:22:43 2020

@author: imyyounge
"""
################################################################################
###                              SET UP                                      ###
################################################################################

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

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X_train, y_train)
print("Regression_intercept")
print(regress.intercept_)
coeff_df = pd.DataFrame(regress.coef_, X_train.columns, columns=['Coefficient'])
coeff_df #Look at the coefficients from the model
coeff_df.to_csv('Linear_reg.csv')

y_pred = regress.predict(X_test) # Make predictions using this model
reg_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Regression_output_head_to_be_printed")
print(reg_output.head(15))

# Evaluating the regression output
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Plot the outcome of the actual and predicted y values
reg_output['Actual-log'] = np.log(reg_output['Actual'])
reg_output['Predicted-log'] = np.log(reg_output['Predicted'])

sns_plot = sns.regplot(x="Actual", y="Predicted", data=reg_output)
fig = sns_plot.get_figure()
plot_file_name = "Actual.vs.prediced.LinearRegression"
fig.savefig(plot_file_name,format='jpeg', dpi=100)

sns_plot1= sns.regplot(x="Actual-log", y="Predicted-log", data=reg_output)
fig = sns_plot1.get_figure()
plot_file_name = "LOGActual.vs.LOGprediced.LinearRegression"
fig.savefig(plot_file_name,format='jpeg', dpi=100)

print("Reached_the_end")