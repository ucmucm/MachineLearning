#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:22:33 2017

@author: Smiker
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'http://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url,sep='\t',header = 0)
df = df.drop('Unnamed: 0',axis = 1)
nsamp = np.array(df[['lcavol','lweight','age','lbph','lbph','svi','lcp','gleason','pgg45']])
y = np.array(df['lpsa'])
ns_train = nsamp // 2
ns_test = nsamp - ns_train
X_tr = 