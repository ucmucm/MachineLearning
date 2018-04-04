#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:34:49 2017

@author: Smiker
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import linear_model, preprocessing

df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                   +'00342/Data_Cortex_Nuclear.xls',index_col=0)

df1 = df.fillna(df.mean())

Gen = df1['Genotype'].values
OriginalName, y = np.unique(Gen, return_inverse = True)

X = df1[df1.columns[0:77]]
Xs = preprocessing.scale(X)

logreg = linear_model.LogisticRegression(C=1)
logreg.fit(Xs, y)

yhat = logreg.predict(Xs)
acc = np.mean(yhat == y)
print("Accuracy on training data = %f" % acc)

logreg_coef = logreg.coef_[0]
plt.stem(logreg_coef)

maxIndex = logreg_coef.argsort()[-2:][::-1]
names = df1.columns[maxIndex]
print('The names of genes that have largest W[i] are %s and %s'%(names[0], names[1]))


from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
nfold = 10
kf = KFold(n_splits=nfold)
prec = []
rec = []
f1 = []
acc = []
XsNew = np.c_[Xs,y]
np.random.shuffle(XsNew)
Xs1 = XsNew[:,:-1]
y1 = XsNew[:,-1]

for train, test in kf.split(Xs):            
    
    Xtr = Xs1[train,:]
    ytr = y1[train]
    Xts = Xs1[test,:]
    yts = y1[test]
       
    logreg.fit(Xtr, ytr)
    yhat = logreg.predict(Xts)
    
    preci,reci,f1i,_= precision_recall_fscore_support(yts,yhat,average='binary') 
    prec.append(preci)
    rec.append(reci)
    f1.append(f1i)
    acci = np.mean(yhat == yts)
    acc.append(acci)

precm = np.mean(prec)
recm = np.mean(rec)
f1m = np.mean(f1)
accm= np.mean(acc)

prec_se = np.std(prec)/np.sqrt(nfold-1)
rec_se = np.std(rec)/np.sqrt(nfold-1)
f1_se = np.std(f1)/np.sqrt(nfold-1)
acc_se = np.std(acc)/np.sqrt(nfold-1)

print('Precision = {0:.4f}, SE={1:.4f}'.format(precm,prec_se))
print('Recall =    {0:.4f}, SE={1:.4f}'.format(recm, rec_se))
print('f1 =        {0:.4f}, SE={1:.4f}'.format(f1m, f1_se))
print('Accuracy =  {0:.4f}, SE={1:.4f}'.format(accm, acc_se))



Class = df1['class'].values
OriginalName, y = np.unique(Class, return_inverse = True)

logreg1 = linear_model.LogisticRegression(C=1)
logreg1.fit(Xs, y)

yhat = logreg1.predict(Xs)
acc = np.mean(yhat == y)
print("Accuracy on training data = %f" % acc)


import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits=nfold)
C = np.zeros([8,8])

XsNew = np.c_[Xs,y]
np.random.shuffle(XsNew)
Xs2 = XsNew[:,:-1]
y2 = XsNew[:,-1]
mean = []

for train, test in kf.split(Xs):            
    Xtr = Xs2[train,:]
    ytr = y2[train]
    Xts = Xs2[test,:]
    yts = y2[test]

    logreg1 = linear_model.LogisticRegression(C=1)
    logreg1.fit(Xtr, ytr)
    yhat = logreg1.predict(Xts)
    CC = confusion_matrix(yts,yhat)
    C += CC
    MeanE = np.mean(yhat == yts)
    mean.append(MeanE)    
 
C = C / C.sum(axis = 1)    
print(np.array_str(C, precision=4, suppress_small=True))
Mean = np.mean(mean)
acc_se = np.std(mean)/np.sqrt(nfold-1)
print('The overall mean and SE of the test error rate are %f and %f'%(Mean,acc_se))

Xtr = XsNew[:,:-1]
ytr = XsNew[:,-1]
logreg = linear_model.LogisticRegression(C=1)
logreg.fit (Xtr, ytr)

plt.figure()
coe = logreg.coef_
firstRow = coe[0,:]
plt.stem(firstRow)

nfold = 10
kf = sklearn.model_selection.KFold(n_splits=nfold,shuffle=True)
model = linear_model.LogisticRegression(penalty = 'l1', C = 1)

nC = 20
Cs = np.logspace(-2,2,nC)

plt.figure()
error = np.zeros((nC,nfold))
for ifold, ind in enumerate(kf.split(Xs2)):
      
    Itr,Its = ind
    X_tr = Xs2[Itr,:]
    y_tr = y2[Itr]
    X_ts = Xs2[Its,:]
    y_ts = y2[Its]
    
    for ia, c in enumerate(Cs):       
        model.C = c
        model.fit(X_tr,y_tr)        
        y_ts_pred = model.predict(X_ts)
        error[ia,ifold] = np.mean(y_ts_pred != y_ts)

error_mean = np.mean(error,axis=1)
error_std = np.std(error,axis = 1)/np.sqrt(nfold - 1)

imin = np.argmin(error_mean)
error_tgt = error_mean[imin] + error_std[imin]
C_min = Cs[imin]

I = np.where(error_mean < error_tgt)[0]
iopt = I[0]
C_opt = Cs[iopt]
print("Optimal C = %f" % C_opt)

Accu = 1 - error_mean[iopt]
print('The accuracy without regularization is %f \n'%Mean+
      'The accuracy with regularization is %f \n'%Accu+
      'So the accuracy with regularization is greater than \n'+
      'the accuracy without regularization')


plt.semilogx(Cs, error_mean)
plt.semilogx(Cs, error_mean + error_std)
plt.semilogx([C_min,C_opt], [error_tgt,error_tgt], 'rs--')
plt.semilogx([C_opt,C_opt], [-0.02,error_mean[iopt]], 'ro--')
plt.ylim([-0.02,0.3])
plt.legend(['Mean Error Rate','Mean Error Rate + Error Rate Std','Error Rate Target','C Opt'],loc='upper left')
plt.xlabel('Cs')
plt.ylabel('Test Error Rate')
plt.grid()
plt.show()

model.C = C_opt
model.fit(Xs2,y2)
plt.figure()
model_coef = model.coef_[0,:]
plt.stem(model_coef)






































