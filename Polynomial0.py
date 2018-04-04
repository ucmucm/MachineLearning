#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 15:29:33 2017

@author: Smiker
"""
#https://github.com/sdrangan/introml/blob/debe36e39894ffbfbb748cf06f1454cfb04562b7/model_sel/polyfit.ipynb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
import numpy.polynomial.polynomial as poly
import sklearn.model_selection

beta = np.array([1,0.5,0.2])
wstd = 0.2
dtrue = len(beta) - 1

nsamp = 40
xdat = np.random.uniform(-1,1,nsamp)

y0 = poly.polyval(xdat,beta)
ydat = y0 + np.random.normal (0,wstd,nsamp)

d = 3
beta_hat = poly.polyfit(xdat,ydat,d)

plt.figure(0)
xp = np.linspace(-1,1,100)
yp = poly.polyval(xp,beta)
yp_hat = poly.polyval(xp,beta_hat)
plt.xlim(-1,1)
plt.ylim(-1,3)
plt.plot(xp,yp,'r-',linewidth=2)
plt.plot(xp,yp_hat,'g-',linewidth=2)

plt.scatter(xdat,ydat)
plt.xlim([-1,1])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend(['True (dtrue=3)', 'Est (d=3)', 'Data'], loc='upper left')

dtest = np.array(range(1,15))
RSStr = []
for d in dtest:
    beta_hat = poly.polyfit(xdat,ydat,d)
    
    yhat = poly.polyval(xdat,beta_hat)
    RSSd = np.mean((yhat-ydat)**2)
    RSStr.append(RSSd)
plt.figure(1)
plt.plot(dtest,RSStr,'o-')
plt.xlabel('Model order')
plt.ylabel('RSS (training)')
plt.grid()

dtest = [1,3,10]
nd = len(dtest)

nplot = 100
xp = np.linspace(-1,1,nplot)
yp = poly.polyval(xp,beta)

yp_hat = np.zeros((nplot,nd))
plt.figure(figsize = (10,5))

plt.figure(2)
for j,d in enumerate(dtest):
    beta_hat = poly.polyfit(xdat,y0,d)
    yp_hat[:,j] = poly.polyval(xp,beta_hat)
    
    plt.subplot(1,nd,j+1)
    plt.xlim(-1,1)
    plt.ylim(-1.5,3.5)
    plt.plot(xp,yp,'r-',linewidth=1)
    plt.scatter(xdat,y0,marker='o',c='r',linewidth=0)
    plt.plot(xp,yp_hat[:,j],'g-',linewidth=3)
    if d < dtrue:
        plt.title('Bias: d= %d' %d)
    else:
        plt.title('No bias: d = %d' %d)
    plt.grid()
    plt.legend(['True','Est','Data (no noise)'],loc='upper left')
    
ntrial = 100
dtest = [1,3,14]
nd = len(dtest)

plt.figure(3)
nplot = 30
xp = np.linspace(-1,1,nplot)
yp = poly.polyval(xp,beta)

yp_hat_mean = np.zeros((nplot,nd))
yp_hat_std = np.zeros((nplot,nd))

for j,d in enumerate(dtest):
    yp_hat = np.zeros((nplot,ntrial))
    
    for it in range(ntrial):
        ydati = y0 + np.random.normal(0,wstd,nsamp)
        
        beta_hat = poly.polyfit(xdat,ydati,d)
        yp_hat[:,it] = poly.polyval(xp,beta_hat)
        
    yp_hat_mean[:,j] = np.mean(yp_hat,axis=1)
    yp_hat_std[:,j] = np.std(yp_hat,axis = 1)
  
plt.figure(figsize=(10,5))
for j,d in enumerate(dtest):
    
    plt.subplot(1,nd,j+1)
    plt.xlim(-1,1)
    plt.ylim(-1.5,3.5)
    plt.plot(xp,yp,'r-',linewidth=1)
    plt.errorbar(xp,yp_hat_mean[:,j],fmt='g-',yerr=yp_hat_std[:,j],linewidth = 1)
    plt.title('d=%d'%d)
    plt.grid()
    plt.legend(['True','Est'],loc='upper left')
    
ntr = nsamp // 2
nts = nsamp - ntr

xtr = xdat[:ntr]
ytr = ydat[:ntr]

xts = xdat[ntr:]
yts = ydat[ntr:]

plt.figure()
xp = np.linspace(-1,1,100)
yp = poly.polyval(xp,beta)
plt.xlim(-1,1)
plt.plot(xp,yp,'r-',linewidth=3)

plt.plot(xtr,ytr,'bo')
plt.plot(xts,yts,'gs')
plt.grid()
plt.legend(['True','Training','Test'],loc='upper left')

plt.figure()
dtest = np.array(range(0,10))
RSStest = []
RSStr = []

for d in dtest:
    beta_hat = poly.polyfit(xtr,ytr,d)
    
    yhat = poly.polyval(xtr,beta_hat)
    RSSd = np.mean((yhat-ytr)**2)
    RSStr.append(RSSd)
    
    yhat = poly.polyval(xts,beta_hat)
    RSSd = np.mean((yhat-yts)**2)
    RSStest.append(RSSd)

plt.plot(dtest,RSStr,'bo-')
plt.plot(dtest,RSStest,'go-')
plt.xlabel('Model order')
plt.ylabel('RSS')
plt.grid()
plt.ylim(0,1)
plt.legend(['Training','Test'],loc='upper right')

imin = np.argmin(RSStest)
print ("Estimated model order = {:d}".format(dtest[imin]))

plt.figure()
nfold = 10
kf = sklearn.model_selection.KFold(n_splits=nfold,shuffle=True)

dtest = np.arange(0,10)
nd = len(dtest)

RSSts = np.zeros((nd,nfold))
for isplit, Ind in enumerate(kf.split(xdat)):
    
    Itr, Its = Ind
    xtr = xdat[Itr]
    ytr = ydat[Itr]
    xts = xdat[Its]
    yts = ydat[Its]
    
    for it,d in enumerate(dtest):
        beta_hat = poly.polyfit(xtr,ytr,d)
        
        yhat=poly.polyval(xts,beta_hat)
        RSSts[it,isplit] = np.mean((yhat-yts)**2)
        
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

plt.plot([dopt,dopt],[0,0.5],'g--')

plt.ylim(0,1)
plt.xlabel('Model order')
plt.ylabel('Test RSS')
plt.grid()

print ("The estimated model order is {:d}".format(dopt))







