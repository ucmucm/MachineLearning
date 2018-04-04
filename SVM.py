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
        
        
def mnist_resize(img):
    """
    Extracts a character from the image, and places in a 28x28 image to match the MNIST format.
    
    Returns:
    img1:  MNIST formatted 28 x 28 size image with the character from img
    box:   A bounding box indicating the locations where the character was found in img.    
    """
    # Image sizes (fixed for now).  To match the MNIST data, the image 
    # will be first resized to 20 x 20.  Then, the image will be placed in center of 28 x 28 box
    # offet by 4 on each side.
    nx_img = 20   
    ny_img = 20
    nx_box = 28   
    ny_box = 28
    offx = 4
    offy = 4
    
    # TODO:  Convert the image to gray scale using the skimage.color.rgb2gray method.
    # bw = ...
    
    bw = skimage.color.rgb2gray(img)
    
    # Threshold the image using OTSU threshold
    thresh = threshold_otsu(bw)
    bw = closing(bw < thresh, square(3)).astype(int)
    
    # Get the regions in the image.
    # This creates a list of regions in the image where the digit possibly is.
    regions = regionprops(bw)

    # TODO:  Find region with the largest area.  You can get the region area from region.area.
    # region_max = ...
     
    area_max = regions[0].area
    region_max = regions[0]
    for region in regions:
        region_area = region.area
        if region_area > area_max:
            area_max = region_area
            region_max = region
            
    
    # Raise an ImgException if no region with area >= 100 was found
    if (area_max < 100):
        raise ImgException("No image found")    
                
    # Get the bounding box of the character from region_max.bbox
    minr, minc, maxr, maxc = region_max.bbox
    box = [minr,minc,maxr,maxc]
    
    # TODO:  Crop the image in bw to the bounding box
    # bw_crop = bw[...]
    
    bw_crop = bw[minr:maxr,minc:maxc]
        
    # TODO:  Resize the cropped image to a 20x20 using the resize command.
    # You will need to use the mode = 'constant' option
    # bw_resize = ...
    
    bw_resize = resize(bw_crop,(20,20),mode = 'constant')
    
    # TODO:  Threshold back to a 0-1 image by comparing the pixels to their mean value
    
    mean = np.mean(threshold_otsu(bw_resize))
    bw_resize = bw_resize > mean
    
    # TODO:  Place extracted 20 x 20 image in larger image 28 x 28
    # img1 = ...
#     img1 = resize(bw_resize,(28,28),mode = 'constant')
    img1 = np.zeros([28,28])
    img1[offy:offy+bw_resize.shape[0],offx:offx+bw_resize.shape[1]] = bw_resize
    
    return img1, box

# Load an image
img = load_img(13,9)

try:
    # Resize the image
    # img1, box = mnist_resize(img)
    
    img1, box = mnist_resize(img)
    
    # TODO:  Plot the original image, img, along with a red box around the captured character.
    # Use the mpatches.Rectangle and ax.add_patch methods to construct the rectangle.

    minr, minc, maxr, maxc = box
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False,
                             edgecolor='red', linewidth=2)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.add_patch(rect)
    ax.set_axis_off()
    plt.show()
    
    # TODO:  Plot the resized 28 x 28 image, img1.  You can use the plt_digit(img1) command 
    plt.subplot(1,2,2)
    plt_digit(img1)
       
except ImgException as e:
    print(e.msg)
    
    
    # Dimensions
import random
nlet = 1000
nrow = 28
ncol = 28
npix = nrow*ncol
Xlet = np.zeros((nlet, npix))

i = 0
while i < nlet:
    # TODO:  Generate a random character and sample    
    # char_ind = random number corresponding to a lowercase letter except 'O' and 'I'
    # samp_ind = random number from 0 to 49
    
    o_ind = 51
    i_ind = 45
    f = list(range(37,i_ind)) + list(range(i_ind+1,o_ind)) + list(range(o_ind+1,63))
    char_ind = random.choice(f)
    samp_ind = np.random.choice(50)
        
    try:
        # TODO:  Load the image with load_img function
        # img = ...
        img = load_img(char_ind, samp_ind)
        
        # TODO:  Reize the image with mnist_resize function
        # img1, box = ...
        img1, box = mnist_resize(img)
        
        
        # TODO:  Store the image in a row of Xlet[i,:] and increment i
        Xlet[i,:] = img1.ravel()
        i += 1
        
        # Print progress
        if (i % 50 == 0):
            print ('images captured = {0:d}'.format(i))
    except ImgException:
        # Skip if image loading or resizing failed
        pass

import pickle

# TODO

with open( "Xlet.p", "wb" ) as fp:
    pickle.dump( Xlet,  fp)
    
with open( "Xlet.p", "rb" ) as fp:
    Xlet = pickle.load(fp)
    
randInd = random.sample(range(70000), 5000)
Xd = Xdigs[randInd]
yd = ydig[randInd]

Xlets = 2*Xlet - 1
ylets = np.empty(1000)
ylets.fill(10)

X = np.vstack((Xd,Xlets))
y = np.hstack((yd,ylets))

from sklearn import svm

# TODO:  Create a classifier: a support vector classifier
# svc = ...

svc = svm.SVC(probability=False,  kernel="rbf", C=2.8, gamma=.0073,verbose=10)

yh = y[:,None]
All = np.hstack((X,yh))
np.random.shuffle(All)
X = All[:,:-1]
y = All[:,-1]

Xtr = X[:5000,:]
ytr = y[:5000]
Xts = X[5000:,:]
yts = y[5000:]

svc.fit(Xtr,ytr)

yhat_ts = svc.predict(Xts)
acc = np.mean(yhat_ts == yts)
print('Accuaracy = {0:f}'.format(acc))

# TODO
from sklearn.metrics import confusion_matrix

C = confusion_matrix(yts,yhat_ts)

# Normalize the confusion matrix
Csum = np.sum(C,1)
C = C / Csum[None,:]

# Print the confusion matrix
print(np.array_str(C, precision=3, suppress_small=True))
plt.imshow(C, interpolation='none')
plt.colorbar()


# TODO

Ierr = np.where((yts != 10) & (yhat_ts == 10))[0]

# nplt = 4 if Ierr.size > 4 else Ierr.size
nplt = (Ierr.size,4)[Ierr.size > 4]
if nplt != 0:
    plt.figure(figsize=(10, 4))
    for i in range(nplt):        
        plt.subplot(1,nplt,i+1)        
        ind = Ierr[i]    
        plt_digit(Xts[ind,:])        
        title = 'true={0:d} est={1:d}'.format(yts[ind].astype(int), yhat_ts[ind].astype(int))
        plt.title(title)
else:
    print ("No such error found")
    
    
Ierr = np.where((yts == 10) & (yhat_ts != 10))[0]

# nplt = 4 if Ierr.size > 4 else Ierr.size
nplt = (Ierr.size,4)[Ierr.size > 4]
if nplt != 0:
    plt.figure(figsize=(10, 4))
    for i in range(nplt):        
        plt.subplot(1,nplt,i+1)        
        ind = Ierr[i]    
        plt_digit(Xts[ind,:])        
        title = 'true={0:d} est={1:d}'.format(yts[ind].astype(int), yhat_ts[ind].astype(int))
        plt.title(title)
else:
    print ("No such error found")
    
Ierr = np.where((yts != yhat_ts) & (yhat_ts < 10) & (yts < 10))[0]

nplt = 4
plt.figure(figsize=(10, 4))
for i in range(nplt):        
    plt.subplot(1,nplt,i+1)        
    ind = Ierr[i]    
    plt_digit(Xts[ind,:])        
    title = 'true={0:d} est={1:d}'.format(yts[ind].astype(int), yhat_ts[ind].astype(int))
    plt.title(title)

