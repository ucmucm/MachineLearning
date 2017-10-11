#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:42:21 2017

@author: Smiker
"""
import numpy as np
import matplotlib.pyplot as plt


y = np.array([0,1,0,1,1])
X = np.array([[30,0],[50,1],[70,1],[80,2],[100,1]])
Donate = (y == 1)
NoDonate = ( y == 0 )

plt.plot(X[Donate,0],X[Donate,1],'g.')
plt.plot(X[NoDonate,0],X[NoDonate,1],'r.')
plt.legend(['Donate','NoDonate'],loc='upper right')
plt.xlabel('Income')
plt.ylabel('Num Websites')


xt = np.random.uniform(0,100,100)
yt = (1/80)*xt + 1/4
plt.plot(xt,yt)


plt.figure()
xtt = np.arange(-1.5,1.5,0.1)
ytt = e ** -xtt
plt.plot(xtt,ytt,'r')
ytt = e ** -(0.5*xtt)
plt.plot(xtt,ytt)
ytt = e ** -(2*xtt)
plt.plot(xtt,ytt)
plt.legend(['y=e^(-x)','y=e^(-0.5x)','y=e^(-2x)'],loc='upper right')
w = np.array([-1/80,1])
b = -1/4


import numpy as np
def gen_rand(X,w,b):
    z = (X * w).sum(axis = 1) + b
    P = 1 / (1 + e ** (-z))
    y = np.zeros(P.size)
    y[np.where(P > 0.5)] = 1
    return y
    

yy = gen_rand(X,w,b)
    