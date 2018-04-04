#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:43:04 2017

@author: Smiker
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets, linear_model, preprocessing


names =[
    't',                                  
    'q1', 'q2', 'q3',                     
    'dq1', 'dq2', 'dq3',                  
    'I1', 'I2', 'I3',                     
    'eps21', 'eps22', 'eps31', 'eps32',   
    'ddq1', 'ddq2', 'ddq3'                
]

df = pd.read_csv('https://raw.githubusercontent.com/sdrangan/introml/d20a19e4d18f858935fd8940b814a56f85bcbc39/mult_lin_reg/exp1.csv',
                 header=None,index_col = 0,names=names,na_values='?')
df.head(6)

y = np.array(df['I2'])
t = np.array(df.index.get_values())
plt.figure(0)
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('I2')

ytrain = y
Xtrain = np.array(df[['q2','dq2','eps21','eps22','eps31','eps32','ddq2']])
regr = linear_model.LinearRegression()
regr.fit(Xtrain,ytrain)
ytrain_pred = regr.predict(Xtrain)

RSS_tr = np.mean((ytrain_pred - ytrain) ** 2 ) / (np.std(ytrain) ** 2)
Rsq_tr = 1 - RSS_tr
print ("RSS_Train = {:f}".format(RSS_tr))
print ("R^2       = {:f}".format(Rsq_tr))

plt.figure(1)
plt.plot(t,ytrain_pred,'r',linewidth=0.5)
plt.plot(t,ytrain,'b',linewidth=0.5)
plt.xlabel('time')
plt.ylabel('I2')
red_patch = mpatches.Patch(color='red',label='Prdicted I2')
blue_patch = mpatches.Patch(color='blue',label='Actual I2')
plt.legend(handles=[red_patch,blue_patch])
plt.grid()

plt.figure(2)
plt.scatter(ytrain,ytrain_pred)
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid()


df2 = pd.read_csv('https://raw.githubusercontent.com/sdrangan/introml/867485fa6c744d1f9bc045eaf432b4a742fff8a0/mult_lin_reg/exp2.csv',
                 header=None,index_col = 0,names=names,na_values='?')
y2 = np.array(df2['I2'])
t2 = np.array(df2.index.get_values())

ytest = y2
Xtest = np.array(df2[['q2','dq2','eps21','eps22','eps31','eps32','ddq2']])
ytest_pred = regr.predict(Xtest)

plt.figure(3)
plt.plot(t2,ytest_pred,'r',linewidth=0.5)
plt.plot(t2,ytest,'b',linewidth=0.5)
plt.xlabel('time')
plt.ylabel('I2')
red_patch = mpatches.Patch(color='red',label='Prdicted I2')
blue_patch = mpatches.Patch(color='blue',label='Actual I2')
plt.legend(handles=[red_patch,blue_patch])
plt.grid()

RSS_test = np.mean((ytest_pred - ytest) ** 2 ) / (np.std(ytest) ** 2)
Rsq_test = 1 - RSS_test
print ("RSS_Test = {:f}".format(RSS_test))
print ("R^2       = {:f}".format(Rsq_test))

plt.figure(4)
plt.scatter(ytest,ytest_pred)
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid()