#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 09:51:52 2020

@author: Harsha Vardhan
"""

#Case Study on mtcars dataset in Python	download data

#Download data
import statsmodels.api as sm
#https://vincentarelbundock.github.io/Rdatasets/datasets.html
dataset_mtcars = sm.datasets.get_rdataset(dataname='mtcars', package='datasets')
dataset_mtcars.data.head()
mtcars = dataset_mtcars.data
#structure

import numpy as np
import pandas as pd

#summary

#print first / last few rows
dataset_mtcars.data.head(5)
dataset_mtcars.data.tail(5)

#print number of rows
dataset_mtcars.data.shape[0]

#number of columns
dataset_mtcars.data.shape[1]

#print names of columns
dataset_mtcars.data.columns     #dataset_mtcars.data.head(0)

#Filter Rows
#cars with cyl=8
dataset_mtcars.data[(dataset_mtcars.data.cyl==8)]

#cars with mpg <= 27
dataset_mtcars.data[(dataset_mtcars.data.mpg<=27)]

#rows match auto tx
#WHAT DOES THIS MEAN? AUTO TX?

#First Row
print(dataset_mtcars.data.iloc[1])

#last Row
print(dataset_mtcars.data.iloc[31])

# 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
print(mtcars.iloc[[0,3,6,24],[0,5,6]])

# first 5 rows and 5th, 6th, 7th columns of data frame
print(mtcars.iloc[:5,4:7])

#rows between 25 and 3rd last
print(mtcars.iloc[25:len(mtcars)-3])
len(mtcars)

#alternative rows and alternative column
print(mtcars.iloc[::2,::2])

#find row with Mazda RX4 Wag and columns cyl, am
mtcars.filter(items=['cyl','am']).filter(regex='Mazda RX4 Wag',axis=0)

#find row betwee Merc 280 and Volvo 142E and columns cyl, am
mtcars.loc['Merc 280':'Volvo 142E'].filter(items=['cyl','am'])

# mpg > 23 or wt < 2
mtcars[(mtcars['mpg']>23) | (mtcars['wt']<2)]

#using lambda for above
temp = filter(lambda mtcars: mtcars)

#with or condition
for i in mtcars:  if (mtcars['mpg']>23 | mtcars ['print (mtcars[i])


#find unique rows of cyl, am, gear
mtcars.filter(items=['cyl','am','gear']).drop_duplicates()


#create new columns: first make a copy of mtcars to mtcars2
import shutil
import csv
pd.DataFrame({}).to_csv("mtcars2.csv")
shutil.copyfile(mtcars,mtcars2)

#keeps other cols and divide displacement by 61

# multiple mpg * 1.5 and save as original column
