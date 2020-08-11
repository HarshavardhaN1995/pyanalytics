#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:54:32 2020

@author: Harsha Vardhan
"""

#Libraries
import numpy as np
from sklearn.linear_model import LinearRegression
help(np)
dir(np)
np?
np

#Data
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
x
x.shape
y=np.array([5,20,14,32,22,38])
y

#Model
model = LinearRegression().fit(x,y)

#Results
r_sq = model.score(x,y)
print(r_sq)                 #coeff of determination; explains the variation in y wrt to x
model.intercept_
model.coef_

y = model.intercept_ + model.coef_*x

#Predict
y_pred = model.predict(x)
y_pred
print(y_pred)
y_pred2 = model.intercept_ + model.coef_*x
print (y_pred2, sep = '\t')

#new values
x_new = np.arange(5).reshape((-1,1))
x_new
y_new = model.predict(x_new)
print(y_new, sep='\t')

#%% Multiple Linear Regression

x = [[0,1],[5,1],[15,2],[25,2],[35,11],[45,15],[55,34],[60,35]]
x
y = [4,5,20,14,32,22,38,43]
y
x, y = np.array(x), np.array(y)
x
y
x.shape, y.shape

#Model and Fit
model = LinearRegression().fit(x,y)
model.score(x,y)            #model.score = R^2
model.intercept_
model.coef_

#Predict

y_pred = model.predict(x)
y_pred
y_pred2 = model.intercept_ + np.sum(model.coef_*x, axis=1)
y_pred2

#New data

x_new = np.arange(10).reshape((-1,2))
x_new
y_new = model.predict(x_new)
y_new

#Statistics models

import statsmodels.api as sm

from statsmodels.tools import add_constant
x = [[0,1], [5,1], [15,2], [25,2], [35,11], [45,15], [55,34], [60,35]]
x
y = [4,5,20,14,32,22,38,43]
y

x = sm.add_constant(x)
x
model3 = sm.OLS(y,x)
model3
results = model3.fit()
results
results.summary()
results.rsquared
results.rsquared_adj
results.params

results.fittedvalues
results.predict(x)

#%%AIC & BIC  
#https://pypi.org/project/RegscorePy
#pip install RegscorePy
import RegscorePy
#aic(y, y_pred, p)
RegscorePy.aic.aic(y=y, y_pred= results.predict(x), p=1)
RegscorePy.bic.bic(y=y, y_pred= results.predict(x), p=1)

#%%
#Topic ---- Dividing Data into Train and Test 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

mtcars = data('mtcars')
mtcars.head()
mtcars.columns
mtcars.dtypes
mtcars.shape

x = np.arange(10).reshape((5,2))
x
y = range(5)
y
list(y)

x_train, y_train, x_test, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
x_train
y_train
x_test
y_test

train_test_split(y, shuffle = True)
train_test_split(y, shuffle = False)

#%%%
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Load dataset.
iris = load_iris()
type(iris)  #Bunch

X, y = iris.data, iris.target
X
y
#these numpy objects, no head; multi-dim matrices
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)
X_train
X_train.shape
X_test
X_test.shape
y_train
y_test
#%% split data into training and test data.- specify train and test size
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state =123)
print("Labels for training and testing data")
print(train_y)
print(test_y)     

#%%%
from sklearn import linear_model as lm
from statsmodels.formula.api import ols
from pydataset import data
mtcars = data('mtcars')
df1 = mtcars[['mpg','wt','hp']]
df1
MTmodel1 = ols("mpg ~ wt + hp", data=df1).fit()
print(MTmodel1.summary())
predictionM1 = MTmodel1.predict()
predictionM1
#%%%
fig= plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(MTmodel1, 'wt', fig=fig)
#%%%
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(MTmodel1, "wt", ax=ax) 
#%%%%
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_ccpr(MTmodel1, "wt", ax=ax)
#%%%
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)
#%%
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)
#%%%
#fig, ax = plt.subplots()
#fig = sm.graphics.plot_fit(MTmodel1, 0, ax=ax)
#----
IV = df1[['wt','hp']].values
IV
DV= df1['mpg'].values
DV
IV_train, IV_test, DV_train, DV_test = train_test_split(IV, DV,test_size=0.2, random_state=123)
IV_train, IV_test, DV_train, DV_test
#from sklearn import linear_model as lm
MTmodel2a = linear_model.LinearRegression()
MTmodel2a.fit(IV_train, DV_train)  #putting data to model
#MTmodel2a.summary()  #no summary in sklearn
MTmodel2a.intercept_
MTmodel2a.coef_
predicted2a = MTmodel2a.predict(IV_test)
predicted2a
DV_test
r2_score(DV_train, MTmodel2a.predict(IV_train))
#The mean squared error
from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(DV_test, predicted2a)
r2_score(DV_test, predicted2a)  #???
#%%%
# what to LM
# Predicting Continuous, Finding relationship between variables
# Steps : load data, split : DV & IV ; Train and test set
# Load the libraries
# create model : function + IV & DV from Train
# see r2, adjst R2, coeff, significant, other model 
# predict : Model + IV_test -> predicted_y
# rmse : predicted_y - actual_y : as less as possible
# R2 ??
# check for assumption - linear, normality, homoscedascity, multi-collinearity, auto-collinearity

#%%%% Links
#https://pythonprogramminglanguage.com/training-and-test-data/

#%%% Links
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

