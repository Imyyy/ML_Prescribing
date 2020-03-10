#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:52:39 2020

@author: imyyounge
"""
import os
import pandas as pd
import numpy as np

gp = pd.read_csv("gpheadings.csv")
people = pd.read_csv("people.csv")
structure = pd.read_csv("structure.csv")
allgp = pd.read_csv("allgp.csv")
pres = pd.read_csv("T201911PDPI_BNFT.csv")

# Rename any column that ends in _x from the merger function
def rename_x(df):
    for col in df:
        if col.endswith('_x'):
            df.rename(columns={col:col.rstrip('_x')}, inplace=True)
            
            # Remove any column that is unnamed in the dataframe 
# Bit of a misnomer
def rename_unname(df):
    for col in df:
        if col.startswith('Unnamed'):
            df.drop(col,axis=1, inplace=True)
            
            
# Drop any column names _y at the end, as this is a duplicate column from the merge unc
def drop_y(df):
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)
#Might try and not run this for a bit, as think it might be dropping too much?
    
rename_unname(gp)
rename_unname(allgp)
allgp.rename(columns={'Postcode':'To_drop', '1974...':'Postcode', 'NA.10':'Setting_all_gp_reference', 'NA.8':'Provider'}, inplace=True)  
    
to_drop = ['To_drop', 'NA', 'NA.1', 'NA.2','NA.3', 'NA.4', 'NA.5', 'NA.6', 'NA.7', 'NA.9', 'NA.11']
allgp.drop(to_drop, axis=1, inplace=True)   

print(list(allgp.columns))
    
final = pd.merge(gp, allgp, how="outer", left_on=["E8..."], right_on=["Organisation_code"])
    
final.drop(['Organisation_code', 'Provider', 'Address_4'], axis=1, inplace=True) 
drop_y(final)
rename_x(final)
rename_unname(final)
final.head()    
    
rename_unname(people)
people.head()    
    
final = pd.merge(final, people, how='outer', left_on=['E8...'], right_on=['CODE'])
final.drop(['PUBLICATION', 'EXTRACT_DATE', 'CODE', 'POSTCODE'], axis=1, inplace=True)
print(list(final.columns) )
print('-'*20)
print(final.shape)    
    
rename_unname(structure)
structure.head()    
    
final = pd.merge(final, structure, how="outer", left_on=["E8..."], right_on=["Organisation_code"])
print(list(final.columns) )
print('-'*20)
print(final.shape)    
    
rename_x(final)
drop_y(final)    

rename_unname(pres)
pres.drop(['PCT'], axis=1, inplace=True)
final = pd.merge(final, pres, how="outer", left_on=["E8..."], right_on=["PRACTICE"])


print(list(final.columns))
print('-'*20)
print(final.shape)

import csv
final.to_csv("Combined_NHS_data.csv")
    
pres = pd.DataFrame(final)
rename_unname(toy) # Drop the extra column labelled unnamed
pres.columns = [x.lower() for x in pres.columns]
print("Calculating the number of nas in the dataset")
print(pres.isna().sum()) 

pres = pres.dropna(thresh = pres.shape[1]-2) 
print(pres.describe())

print("Calculating the number of nas in the dataset after removing those with NAs")
print(pres.isna().sum()) 

final.drop(["date_close"], axis=1, inplace = True)

# Dealing with categorical columns
pres.drop(["address_2", "status_code", "subtype", "organisation_code", "postcode", "primary_care_organisation_type", "address_3", "bnf.chemical", "bnf.letters", "bnf.code", "practice", "ons_ccg_code", "sex", "age"], axis=1, inplace = True) 

final = pd.DataFrame(pres)

final['ccg_code1'] = final.groupby('ccg_code').ngroup()
final['high_level_health_geography1'] = final.groupby('high_level_health_geography').ngroup()
final['commissioner1'] = final.groupby('commissioner').ngroup()
final['sha1'] = final.groupby('sha').ngroup()
final['bnf.name1'] = final.groupby('bnf.name').ngroup()
final['e8...1'] = final.groupby('e8...').ngroup()

#Then drop the old columns
final.drop(["ccg_code", "type", "ccg/pct", "high_level_health_geography", "sha", "bnf.name", "commissioner", 'e8...'], axis=1, inplace = True)

# One hot encoding
print(final.shape)
print('-'*20)
final = pd.concat([final, pd.get_dummies(final['area'])], 1) # Trying this tactic with status_code
print(final.shape)
print('-'*20)
final = pd.concat([final, pd.get_dummies(final['national_grouping'])], 1)
print('-'*20)
print(list(final.columns))  # Can see all the column names added on
final.drop(["area", "national_grouping"], axis=1, inplace = True)

final['bnf.chapter'] = pd.to_numeric(final['bnf.chapter'])

print(final.info()) 

import csv 
final.to_csv("final.csv")



    