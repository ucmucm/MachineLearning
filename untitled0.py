#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:09:20 2017

@author: Smiker
"""

import numpy as np
import matplotlib.pyplot as plt




def yy(x):
    return 0.5 - 4*x + 4*(x**3)


alpha = 0.01
num_iters = 400
x = 1

def gradientDescent(yy,alpha,x):
    return x - alpha*(yy(x))
    
for i in range(0,num_iters):
    x = gradientDescent(yy,alpha,x)

