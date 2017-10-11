#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:18:13 2017

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

#eachYMean = np.mean((Xtr-ytr_pred)**2,)
#eachYstd = np.std(Xtr,axis = 1)

ym = np.mean(y0)
syy = np.mean((y0-ym)**2)
#Rsq = np.zeros(nneuron)
#for k in range(nneuron):
#    Xm = np.mean(X0[:,k])
#    sxy = np.mean((X0[:,k]-Xm)*(y0-ym))
#    sxx = np.mean((X0[:,k]-Xm)**2)
#    Rsq[k] = (sxy)**2/sxx/syy
    
#    print ("{0:2d} Rsq = {1:f}".format(k,Rsq[k]))

Xm = np.mean(X0,axis = 0)
X_demean = X0 - Xm[None,:]
sxy = np.mean(X_demean * (y0[:,None]-ym),axis = 0)
sxx = np.mean(X_demean **2,axis = 0)
Rsq = (sxy)**2/sxx/syy

plt.stem(Rsq)

d = 100
Rsq_sort = np.argsort(Rsq)
rsq_sort = np.flipud(Rsq_sort)[1:]
#Isel = Rsq_sort[nneuron-d-1:nneuron-1]
Isel = rsq_sort[:d]
print ("The neurons with the tem highest R^2 values = ",end='')
print (','.join(str(x) for x in Rsq_sort[nneuron-11:nneuron-1]))

Xtr = X0[:ntr,Isel]
ytr = y0[:ntr]
Xts = X0[ntr:,Isel]
yts = y0[ntr:]

regr = sklearn.linear_model.LinearRegression()
regr.fit(Xtr,ytr)
yts_pred = regr.predict(Xts)
RSS_per = np.mean((yts_pred-yts)**2)
RSS_normalized_test = np.mean((yts_pred - yts) ** 2 ) / (np.std(yts) ** 2)
print ("Test RSS per sample = {:f}".format(RSS_per))
print ("Normalized test RSS = {:f}".format(RSS_normalized_test))

plt.figure()
plt.scatter(yts,yts_pred)
plt.plot([-0.3,0.3],[-0.2,0.2],'r')

#Isel = Rsq_sort[nneuron-d-1:nneuron-1]
#X0 = X0[:,Isel]
plt.figure()
nfold = 10
kf = sklearn.model_selection.KFold(n_splits=nfold,shuffle=True)

dtest = np.arange(10,200,10)
nd = len(dtest)

RSSts = np.zeros((nd,nfold))

for it,d in enumerate(dtest):    
    Isel = rsq_sort[:d]
    #Isel = Rsq_sort[nneuron-d-1:nneuron-1]
    Xtem = X0[:,Isel]
   
    for isplit, Ind in enumerate(kf.split(Xtem)):
#        print ("Processing for folder %d with order %d"%(isplit,d))
        Itr, Its = Ind
        xtr = Xtem[Itr,:]
        ytr = y0[Itr]
        xts = Xtem[Its,:]
        yts = y0[Its]
        
        regr = sklearn.linear_model.LinearRegression()
        regr.fit(xtr,ytr)
        yts_pred = regr.predict(xts)
        RSSts[it,isplit] = np.mean((yts_pred - yts)**2)*yts_pred.size
    
RSS_mean = np.mean(RSSts,axis=1)
RSS_std = np.std(RSSts,axis = 1)/np.sqrt(nfold-1)
plt.errorbar(dtest,RSS_mean,yerr=RSS_std,fmt='-')
plt.xlabel('Model order')
plt.ylabel('Test RSS')
plt.grid()

imin = np.argmin(RSS_mean)
print ("The selected model order is {:d}".format(dtest[imin]))

plt.figure()
imin = np.argmin(RSS_mean)
RSS_tgt = RSS_mean[imin] + RSS_std[imin]

I = np.where(RSS_mean <= RSS_tgt)[0]
iopt = I[0]
dopt = dtest[iopt]

plt.errorbar(dtest,RSS_mean,yerr=RSS_std,fmt='o-')
plt.plot([dtest[0],dtest[imin]],[RSS_tgt,RSS_tgt],'--')
#lt.plot([dopt,dopt],[0.0014,0.0020],'g--')
plt.xlabel('Model order')
plt.ylabel('Test RSS')
plt.grid()

print ("The estimated model order is {:d}".format(dopt))

Xs = preprocessing.scale(X0)

nfold = 10
kf = sklearn.model_selection.KFold(n_splits=nfold,shuffle=True)
model = sklearn.linear_model.Lasso(warm_start = True)

nalpha = 100
alphas = np.logspace(-5,-1,nalpha)
RSSts_lasso = np.zeros((nalpha,nfold))

for ifold, Ind in enumerate(kf.split(Xs)):
    Itr, Its = Ind
    X_tr = Xs[Itr,:]
    y_tr = y0[Itr]
    X_ts = Xs[Its,:]
    y_ts = y0[Its]
    
    for ia,a in enumerate(alphas):
        
        model.alpha = a
        model.fit(X_tr,y_tr)
        y_ts_pred = model.predict(X_ts)
        RSSts_lasso[ia,ifold] = np.mean((y_ts_pred - y_ts) ** 2)

plt.figure()
RSSts_mean_lasso = np.mean(RSSts_lasso,axis = 1)
RSSts_std_lasso = np.std(RSSts_lasso,axis = 1)/np.sqrt(nfold - 1)
plt.errorbar(alphas,RSSts_mean_lasso,yerr = RSSts_std_lasso,fmt='-')
plt.xlabel('Alpha')
plt.ylabel('Test RSS')
plt.grid()

imina = np.argmin(RSSts_mean_lasso)

RSS_tgt_lasso = RSSts_mean_lasso[imina] + RSSts_std_lasso[imina]
I = np.where(RSSts_mean_lasso <= RSS_tgt_lasso)[0]
iopt = I[-1]
alpha_opt = alphas[iopt]
print ("The optimal alpha is %f" % alpha_opt)

plt.figure()
model.alpha = alpha_opt
model.fit(X0,y0)
y_opt = model.predict(X0)
plt.scatter(y0,y_opt)
plt.plot([-0.3,0.3],[-0.2,0.2],'r')






























