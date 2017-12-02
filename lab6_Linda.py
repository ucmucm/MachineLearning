# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:37:48 2017

@author: Yingda
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from sklearn import datasets, linear_model, preprocessing

mnist = fetch_mldata("MNIST original")
    
Xdig = mnist.data
ydig = mnist.target

Xdigs = Xdig.astype(float)/255*2-1

nplt = 4
nsamp = Xdig.shape[0]
Iperm = np.random.permutation(nsamp)

def plt_digit(x):
    nrow = 28
    ncol = 28
    xsq = x.reshape((nrow,ncol))
    plt.imshow(xsq,  cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])

for i in range(nplt):
    ind = Iperm[i]
    plt.subplot(1,nplt,i+1)
    plt_digit(Xdigs[ind,:])
    plt.title(ydig[ind])

class ImgException(Exception):
    def __init__(self, msg='No msg'):
        self.msg = msg

import matplotlib.image as mpimg
import skimage.io
from skimage.filters import threshold_mean
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
    if char_ind >= 10:
        Charind = "0" + str(char_ind)
    else: 
        Charind = "00" + str(char_ind)
    
    if samp_ind >= 10:
        Sampind = "0" + str(samp_ind)
    else: 
        Sampind = "00" + str(samp_ind)
    
    # TODO:  Set the file name based on char_ind and samp_ind
    # fname = ...
    fname = "EnglishHnd/English/Hnd/Img/Sample"+Charind+"/img"+Charind+"-"+Sampind+".png"
    
    # TODO:  Use the os.path.isfile command to check if the file exists.  
    # If not raise an ImgException with the message "[fname] not found"
    

    # TODO:  Use the skimage.io.imread() command to read the png file and return the image.
    # img = ...
    if(not os.path.isfile(fname)):
        raise ImgException("File not found")
    
    img = skimage.io.imread(fname)
    
    return img

char_ind = 47
samp_inds = [6,70]
#samp_inds = [6]
for samp_ind in samp_inds:
    try:
        img = load_img(char_ind=char_ind, samp_ind=samp_ind)
        print("Char = %d samp=%d" % (char_ind, samp_ind))
        plt.figure()
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
    
    #bw = clear_border(bw)
    
    # Get the regions in the image.
    # This creates a list of regions in the image where the digit possibly is.
    regions = regionprops(bw)

    # TODO:  Find region with the largest area.  You can get the region area from region.area.
    # region_max = ...
    areas = []
    #area_max = regions[0].area
    for region in regions:
        #area_max = np.max(area_max,region.area)
        areas.append(region.area)
    area_max = np.max(areas)
    
    region_max = regions[0]
    for region in regions:
        if(region.area == area_max):
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
    bw_resize = resize(bw_crop, (nx_img, ny_img), mode='constant')
    
    # TODO:  Threshold back to a 0-1 image by comparing the pixels to their mean value
    thresh = threshold_mean(bw_resize)
    binary = bw_resize > thresh
    
    
    # TODO:  Place extracted 20 x 20 image in larger image 28 x 28
    # img1 = ...
    img1 = np.zeros((nx_box,ny_box)).astype(int)
    img1[offy:offy+binary.shape[0],offx:offx+binary.shape[1]] = binary
    return img1, box

# Load an image
img = load_img(13,9)

try:
    # Resize the image
    img1, box = mnist_resize(img)
    
    # TODO:  Plot the original image, img, along with a red box around the captured character.
    # Use the mpatches.Rectangle and ax.add_patch methods to construct the rectangle.
    rect = mpatches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0],
                                  fill=False, edgecolor='red', linewidth=2)
    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.add_patch(rect)
    
    # TODO:  Plot the resized 28 x 28 image, img1.  You can use the plt_digit(img1) command 
    plt.figure()
    plt_digit(img1)
       
except ImgException as e:
    print(e.msg)





# Dimensions
nlet = 1000
nrow = 28
ncol = 28
npix = nrow*ncol
Xlet = np.zeros((nlet, npix))
ylet = np.zeros(nlet)

i = 0
while i < nlet:
    # TODO:  Generate a random character and sample    
    # char_ind = random number corresponding to a lowercase letter except 'O' and 'I'
    # samp_ind = random number from 0 to 49
    char_ind = np.random.randint(37,63)
    
    while(char_ind == 51 or char_ind == 45):
        char_ind = np.random.randint(37,63)
    
    samp_ind = np.random.randint(0,50)
  
        
    try:
        # TODO:  Load the image with load_img function
        # img = ...
        img = load_img(char_ind, samp_ind)
        
        # TODO:  Reize the image with mnist_resize function
        # img1, box = ...
        img1, box = mnist_resize(img)
        
        # TODO:  Store the image in a row of Xlet[i,:] and increment i
        Xlet[i,:] = img1.reshape((1,npix))
        ylet[i] = 10
        i += 1
        
        # Print progress
        if (i % 50 == 0):
            print ('images captured = {0:d}'.format(i))
    except ImgException:
        # Skip if image loading or resizing failed
        pass


with open( "Xlet.p", "wb" ) as fp:
    pickle.dump(Xlet,  fp)

with open( "ylet.p", "wb" ) as fp:
    pickle.dump(ylet,  fp)
    
with open("Xlet.p", "rb") as fp:
    Xlet = pickle.load(fp)
    
with open("ylet.p", "rb") as fp:
    ylet = pickle.load(fp)



ndig = 5000
Xdig = np.zeros((ndig, npix))
ydig = np.zeros(ndig)

i = 0
while i < ndig:
    # TODO:  Generate a random character and sample    
    # char_ind = random number corresponding to a lowercase letter except 'O' and 'I'
    # samp_ind = random number from 0 to 49
    dig_ind = np.random.randint(1,11)
    
    samp_ind = np.random.randint(0,50)
  
        
    try:
        # TODO:  Load the image with load_img function
        # img = ...
        img = load_img(dig_ind, samp_ind)
        
        # TODO:  Reize the image with mnist_resize function
        # img1, box = ...
        img1, box = mnist_resize(img)
        
        # TODO:  Store the image in a row of Xlet[i,:] and increment i
        Xdig[i,:] = img1.reshape((1,npix))
        ydig[i] = dig_ind - 1
        i += 1
        
        # Print progress
        if (i % 50 == 0):
            print ('images captured = {0:d}'.format(i))
    except ImgException:
        # Skip if image loading or resizing failed
        pass


with open( "Xdig.p", "wb" ) as fp:
    pickle.dump(Xdig,  fp)
    
with open("Xdig.p", "rb") as fp:
    Xdig = pickle.load(fp)

with open( "ydig.p", "wb" ) as fp:
    pickle.dump(ydig,  fp)

with open("ydig.p", "rb") as fp:
    ydig = pickle.load(fp)

X = np.vstack((Xlet,Xdig))
y = np.hstack((ylet,ydig))

X = 2 * X - 1 



from sklearn import svm
svc = svm.SVC(probability=False,  kernel="rbf", C=2.8, gamma=.0073,verbose=10)

nsamp = X.shape[0]
Iperm = np.random.permutation(nsamp)

ntr = 5000

Xtr = X[Iperm[:ntr],:]
ytr = y[Iperm[:ntr]]
Xts = X[Iperm[ntr:],:]
yts = y[Iperm[ntr:]]

svc.fit(Xtr,ytr)

with open( "svc.p", "wb" ) as fp:
    pickle.dump( [svc, X, y, Iperm, ntr], fp)

yhat_ts = svc.predict(Xts)
acc = np.mean(yhat_ts == yts)
print('Accuaracy = {0:f}'.format(acc))

with open("svc_test.p", "wb") as fp:
    pickle.dump([yts,yhat_ts,Xts], fp)

with open("svc_test.p", "rb") as fp:
    yts,yhat1,Xts = pickle.load(fp)

Ierr = np.where((yhat_ts != yts))[0]
