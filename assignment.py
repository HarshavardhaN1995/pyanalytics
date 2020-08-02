#Denco assignment
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 01:08:00 2020

@author: Harsha Vardhan
"""

import numpy as np
import pandas as pd

datadenco = pd.read_csv('denco.csv')
datadenco.head()

#loyal customers
a = datadenco.custname.value_counts('')
a
dir(a)
a.sort_values(ascending=False).head(5)

#most revenue contributing customers
b = datadenco.groupby('custname').revenue.sum()
b
b.sort_values(ascending=False).head(5)

#part numbers contributing most revenue

c = datadenco.groupby('partnum').revenue.sum().sort_values(ascending=False).head(5)
c

#parts having highest profit margin
d = datadenco.groupby('partnum').margin.sum().sort_values(ascending=False).head(5)
d

#top buying customers = most revenue providing customers
datadenco.groupby('custname').revenue.sum().sort_values(ascending=False).head(5)
