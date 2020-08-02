#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:28:46 2020

@author: rajupillivenkata
"""


#%%

a=[1,2,3]
a
print(a)
type(a)

#%%

b=[1,'c',a]
b

d=[bool(1),'c']
print(d)

e=[bool(True),'c']
e

#%%


t=(1,2,[3,4,5])
type(t)
print(t)

#%%

dic={1:'Xhose', 2:a, 3:bool(b), 'year':1920}
print(dic)
type(dic)

dic[1]
dic.get(2)
dir(dic)

#%%

s1={'Ralph',"Fiennes","""Ralph"""}
print(s1)
sorted(s1)

s2={1,5,7,9}
s2.union(s1)
sorted(s2.union(s1))  #will give error because of str and int

dict #strange that data types can be used as variables as well

#%%

str1='Kunkka'
len(str1)
type(str1)

#%%

for i in a: 
    print(i)
    
 for i in a: print(b,i+3)    #take note of this
 for i in a: print(a[i-1]+1)
 
 #%%
 
 import numpy
 import numpy as np

a=np.array([1,2,3])
b=np.array([(1.5,2,3),(4,5,6)],dtype=float)
c=np.array([[(1.5,2,3),(4,5,6)],[(3,2,1),(4,5,6)]],dtype=float)
print(a)
print(b)
print(c)

np.zeros((3,4))
np.ones((2,3,4),dtype=np.int16)
d=np.arange(10,25,5)
print(d)
np.linspace(0,2,9)
x=np.arange(1,100000,2)
print(x)
x.mean()
x.sum()
x[1:100]
x[1:50:10]
x.shape

#%%

import pandas as pd

df1=pd.DataFrame({'rollno':[1,2,3,4], 'name':["Dhiraj","Kounal","Akhil","Pooja"], 'marks':[40,50,60,70], 'gender':['M','M','M','F']})
df1
type(df1)
df1.columns       #column names
df1.describe()    #various numerical values like mean, deviation etc
df1.dtypes        #data types
df1.shape         #number of rows and columns
df1.groupby('gender').size()
df1.groupby('gender')
df1.groupby('gender')['marks'].mean()
df1.groupby('gender').aggregate({'marks':[np.mean,'max','min','std','count']})

#%%

import matplotlib.pyplot as plt
#https://matplotlib.org
df1.groupby('gender').size()
df1.groupby('gender').size().plot(kind='bar')

plt.hist(df1['marks'])

#https://seaborn.pydata.org/index.html

import seaborn as sns

#sns.net(style="ticks", color_codes=True)

iris = sns.load_dataset("iris")
iris.head()
iris.tail()
df1.groupby('gender').size()
iris.groupby('species').size().plot(kind='bar')
sns.pairplot(iris)

#%%

#Load Inbuilt datasets
import statsmodels.api as sm
#https://vincentarelbundock.github.io/Rdatasets/datasets.html
mtcars=sm.datasets.get_rdataset(dataname='mtcars', package='datasets')
mtcars.data.head()
mtcars.data.tail()

#%%

#Load from Excel/CSV and export to
data = mtcars.data
data.head(6)
type(data)
data.to_csv('mtcars.csv')
data.to_excel('mtcarsExcel.xlsx','sheet3',header=True)

#writing to multiple sheets
writer = pd.ExcelWriter('test.xlsx',engine='xlsxwriter')
#write each dataframe to a different worksheet
data.to_excel(writer, sheet_name='sheet1')
data.to_excel(writer, sheet_name='sheet2')
#close the Pandas Excel writer and output the Excel file
writer.save()









