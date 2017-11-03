#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:49:54 2017

@author: Smiker
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
Xdig = mnist.data
ydig = mnist.target

Xdigs = Xdig/np.max(Xdig)


def plt_digit(x):
    nrow = 28
    ncol = 28
    xsq = x.reshape((nrow,ncol))
    plt.imshow(xsq,  cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])

# Select random digits
nplt = 4
nsamp = Xdigs.shape[0]
Iperm = np.random.permutation(nsamp)

# Plot the images using the subplot command
for i in range(nplt):
    ind = Iperm[i]
    plt.subplot(1,nplt,i+1)
    plt_digit(Xdigs[ind,:])
    
class ImgException(Exception):
    def __init__(self, msg='No msg'):
        self.msg = msg
        
import matplotlib.image as mpimg
import skimage.io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.transform import resize
import matplotlib.patches as mpatches
from skimage import data
import skimage

import os.path

def load_img(char_ind, samp_ind):
    """
    Returns the image from the dataset given a character and sample index.
    
        
    If the file doesn't exist, it raises an Exception with the filename.   
    """ 
    
    # TODO:  Set the file name based on char_ind and samp_ind
    # fname = ...
    
    dname = "Sample0" + str(char_ind)
    if samp_ind < 10:
        fname = "img0" + str(char_ind) + "-00" + str(samp_ind) + ".png"
    else:
        fname = "img0" + str(char_ind) + "-0" + str(samp_ind) + ".png"
    
    
    
    # TODO:  Use the os.path.isfile command to check if the file exists.  
    # If not raise an ImgException with the message "[fname] not found"
    
    if not os.path.isfile("English/Hnd/Img/" + dname +"/"+ fname):
        raise ImgException("File not found")

    # TODO:  Use the skimage.io.imread() command to read the png file and return the image.
    # img = ...
    img = skimage.io.imread("English/Hnd/Img/" + dname +"/"+fname)

    return img

char_ind = 47
samp_inds = [6,70]
for samp_ind in samp_inds:
    try:
        img = load_img(char_ind=char_ind, samp_ind=samp_ind)
        print("Char = %d samp=%d" % (char_ind, samp_ind))
        plt.imshow(img)
    except ImgException as e:
        print(e.msg)