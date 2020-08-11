#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:55:35 2020

@author: Harsha 
"""


#%% Linear Regression -1 Marketing Data - Sales - YT, FB, print
#libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model #1st method
import statsmodels.api as sm  #2nd method
import matplotlib.pyplot as plt
import seaborn as sns

url ='https://raw.githubusercontent.com/DUanalytics/datasets/master/R/marketing.csv'
marketing = pd.read_csv(url)
marketing.head()

#describe data
marketing.describe()

#visualise few plots to check correlation
plt.scatter(marketing.youtube,marketing.sales, color = 'red')
plt.xlabel("youtube"), plt.ylabel("sales")

plt.scatter(marketing.facebook,marketing.sales,color = 'green')
plt.xlabel("facebook"), plt.ylabel("sales")

plt.scatter(marketing.newspaper,marketing.sales, color = 'blue')
plt.xlabel("newspaper"), plt.ylabel("sales")

#split data into train and test
x = marketing[["youtube","facebook","newspaper"]]
y = marketing[["sales"]].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 42)

#build the model
model = linear_model.LinearRegression().fit(x_train,y_train)    #using 1st method
model.score(x_train,y_train)
model.coef_
model.intercept_

#predict on test values
y_pred = model.predict(x_test)
y_pred

#find metrics - R2, Adjt R2, RMSE, MAPE etc
x_train=sm.add_constant(x_train)
x_train
model2 = sm.OLS(y_train,x_train)
results = model2.fit()
results.summary()

results.rsquared
results.rsquared_adj
mean_squared_error(y_test,y_pred)

#predict on new value
newdata = pd.DataFrame({'youtube':[50,60,70], 'facebook':[20, 30, 40], 'newspaper':[70,75,80]})
newdata
#your ans should be close to [ 9.51, 11.85, 14.18] 
y_pred = model.predict(newdata)
y_pred
