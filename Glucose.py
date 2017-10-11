'''
                    _ooOoo_
                   o8888888o
                   88" . "88
                   (| -_- |)
                   O\  =  /O
                ____/`---'\____
              .'  \\|     |//  `.
             /  \\|||  :  |||//  \
            /  _||||| -:- |||||-  \
            |   | \\\  -  /// |   |
            | \_|  ''\---/''  |   |
            \  .-\__  `-`  ___/-. /
          ___`. .'  /--.--\  `. . __
       ."" '<  `.___\_<|>_/___.'  >'"".
      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
      \  \ `-.   \_ __\ /__ _/   .-` /  /
 ======`-.____`-.___\_____/___.-`____.-'======
                    `=---='
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                In God We Trust
                 No Bug At All
 =============================================
 
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
import pandas as pd

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
nsamp, natt = X.shape
print ("num samples = {:d} num attributes = {:d}".format(nsamp,natt))

ym = np.mean(y)
syy = np.mean((y-ym)**2)
Rsq =  np.zeros(natt)
beta0 = np.zeros(natt)
beta1 = np.zeros(natt)
for k in range(natt):
    xm = np.mean(X[:,k])
    sxy = np.mean((X[:,k]-xm)*(y-ym))
    sxx = np.mean((X[:,k]-xm)**2)
    beta1[k] = sxy/sxx
    beta0[k] = ym - beta1[k]*xm
    Rsq[k] = (sxy) ** 2 /sxx/syy
    print ("{:2d} Rsq={:f}".format(k,Rsq[k]))

imax = np.argmax(Rsq)
xmin = np.min(X[:,imax])
xmax = np.max(X[:,imax])
ymin = beta0[imax] + beta1[imax]*xmin
ymax = beta0[imax] + beta1[imax]*xmax

ym = np.mean(y)
y1 = y-ym
Xm = np.mean(X,axis = 0)
X1 = X - Xm[None,:]

syy = np.mean(y1 ** 2)
Sxx = np.mean(X1 ** 2, axis = 0)
Sxy = np.mean(X1 * y1[:,None],axis = 0)

beta1 = Sxy/Sxx
beta0 = ym - beta1*Xm
Rsq = Sxy**2/Sxx/syy

ns_train = 300
ns_test = nsamp - ns_train
X_tr = X[:ns_train,:]
y_tr = y[:ns_train]

regr = linear_model.LinearRegression()
regr.fit(X_tr,y_tr)
print (regr.coef_)

y_tr_pred = regr.predict(X_tr)
RSS_tr = np.mean((y_tr_pred-y_tr) ** 2)/(np.std(y_tr)**2)
Rsq_tr = 1-RSS_tr
print ("\nRSS per sample = {:f}".format(RSS_tr))
print ("R^2 =            {:f}\n".format(Rsq_tr))

#plt.scatter(y_tr,y_tr_pred)
#plt.plot([0,350],[0,350],'r')
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.grid()

X_test = X[ns_train:,:]
y_test = y[ns_train:]
y_test_pred = regr.predict(X_test)
RSS_test = np.mean((y_test_pred - y_test) ** 2)/(np.std(y_test)**2)
Rsq_test = 1 - RSS_test
print("RSS per sample = {:f}".format(RSS_test))
print ("R^2           = {:f}".format(Rsq_test))

plt.scatter(y_test,y_test_pred)
plt.plot([0,350],[0,350],'r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid()

ones = np.ones((ns_train,1))
A = np.hstack((ones,X_tr))
print (A.shape)

out = np.linalg.lstsq(A,y_tr)
beta = out[0]
print (beta)
print (regr.coef_)
print (regr.intercept_)