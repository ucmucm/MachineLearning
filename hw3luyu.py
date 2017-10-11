#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:06:59 2017

@author: Smiker
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.io
mat_dict = scipy.io.loadmat('StevensonV2.mat')

#TODO
var=mat_dict.keys()
X0 = mat_dict['spikes'].T
y0 = mat_dict['handVel'][0,:]

X0.shape
row=X0.shape[0]
column=X0.shape[1]
print("nt is {0:d}".format(row))
print("nneuron is {0:d}".format(column))

time=mat_dict['time'].reshape(15536)

time_roll=np.roll(time,1)
res=time-time_roll
res=np.delete(res,0,0)
tsamp=np.mean(res)
ttotal=np.sum(res)
print("tsamp is {0:f}".format(tsamp))
print("ttotal is {0:f}".format(ttotal))

Xtr=X0[0:row//2,:]
ytr=y0[0:row//2]
Xts=X0[row//2:,:]
yts=y0[row//2:]

import sklearn.linear_model

# TODO
regr=sklearn.linear_model.LinearRegression()
regr.fit(Xtr,ytr)

yts_pred=regr.predict(Xts)
RSS_test=np.mean((yts_pred-yts)**2)/(np.std(yts)**2)
print("RSS per sample = {0:f}".format(RSS_test))