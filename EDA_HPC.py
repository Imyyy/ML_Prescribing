#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:09:32 2020

@author: imyyounge
"""
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set(style="whitegrid") # SNS set up
final = pd.read_csv("final.csv") 

##########################################################################################
                                        # EDA
##########################################################################################

# Small dataset for correlation matrix
small = final.iloc[:, : 17] # Select all the columns that are not from one hot encoding
small['logitems'] = np.log(small['items'])
small['logquantity'] = np.log(small['quantity'])

# Geography and number of patients
plt.figure(figsize=(16, 6))
cmap = sns.diverging_palette(15, 220, as_cmap=True, center="light", s = 99)

sns_plot = sns.boxplot(y=small['number_of_patients'], x=small["high_level_health_geography1"], palette="Blues")
plot_file_name="numpatients-highlvelhealthgeog.jpg"
fig = sns_plot.get_figure()
fig.savefig(plot_file_name,format='jpeg', dpi=100)


# Number of items
b = sns.jointplot(x="bnf.chapter", y="items", data=small)
plot_file_name="bnfchapter-items.jpg" 
fig = b.get_figure()
fig.savefig(plot_file_name,format='jpeg', dpi=100)

c = sns.jointplot(x="number_of_patients", y="logitems", data=small)
plot_file_name="numberpatients-logitems.jpg" 
fig = c.get_figure()
fig.savefig(plot_file_name,format='jpeg', dpi=100)

# Count plot of the number of SHAs
d = sns.countplot(x='sha1', data=small) # There are 28 SHA's 
plot_file_name="countplotnumberSHAs.jpg" 
fig = d.get_figure()
fig.savefig(plot_file_name,format='jpeg', dpi=100)

# Ran up to here without any errors in the console

# CORRELATION MATRIX - think this will give an error but lets see how we go
corr = small.corr() # Compute the correlation matrix
mask = np.triu(np.ones_like(corr, dtype=np.bool)) # Generate a mask for the upper triangle
f, ax = plt.subplots(figsize=(15, 13)) # Set up the matplotlib figure
cmap = sns.diverging_palette(15, 220, as_cmap=True, center="light", s = 99) # Generate a custom diverging colormap, have changed so red is negative
    # center = whether white or black in middle of colour range
    # s : saturation. Integer 1 - 100
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}
           ) # Draw the heatmap with the mask and correct aspect ratio
list_labels = ('Date practice opened', 'Number of patients', 'Date of joining organisation', 'Date of leaving parent organisation',
               'Number of items', 'Net ingredient cost', 'Actual cost', 'Quantity given in prescriptions', 
               'BNF chapter', 'BNF section', 'BNF paragraph', 'CCG', 'High level health geography', 'Commissioner', 'SHA', 
               'BNF Name', 'Practice identifier', 'log items', 'log quantity')
plt.xticks(np.arange(19), list_labels) # Change the x axis labels to actual names
plt.yticks(np.arange(19), list_labels) # Change the y axis labels
plot_file_name="Correlation_matrix.jpg" 
plt.savefig(plot_file_name,format='jpeg', dpi=100)


##########################################################################################
                                        # PLOTS
##########################################################################################

