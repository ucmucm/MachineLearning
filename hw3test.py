#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:00:41 2017

@author: Smiker
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import sklearn.linear_model
import sklearn.model_selection
import numpy.polynomial.polynomial as poly
from sklearn import preprocessing


mat_dict = scipy.io.loadmat('StevensonV2.mat')

mat_dict.keys()

X0 = mat_dict['spikes'].T
y0 = mat_dict['handVel'][0,:]

nt,nneuron = np.shape(X0)

time = mat_dict['time'].T

tsamp = (time[1] - time[0])[0]
ttotal = (time[nt-1] - time[0])[0]

ntr = (int) (nt // 2)
nts = nt - ntr

Xtr = X0[:ntr,:]
ytr = y0[:ntr]
Xts = X0[ntr:,:]
yts = y0[ntr:]

regr = sklearn.linear_model.LinearRegression()
regr.fit(Xtr,ytr)
yts_pred = regr.predict(Xts)
RSS_ts = np.mean((yts_pred - yts) ** 2 ) / (np.std(yts) ** 2)
print ("RSS_Test = {:f}".format(RSS_ts))