#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:35:56 2017

@author: Smiker
"""

from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

sr, y = read('viola.wav')

y = y.astype(float)

print ("The sample rate is %d Hz" %sr)
print ("The number of samples is %d" %(y.size))
print ("The file length is %f seconds"%((y.size)/sr))



import IPython.display as ipd
ipd.Audio(y, rate=sr) # load a NumPy array


scale = np.sqrt(np.mean(y**2))
y =  y / scale


nfft = 1024


nframe = (int)(y.size / nfft) + 1
y_t = np.zeros(nframe*nfft)
y_t[:y.size]=y
yframe = y_t.reshape([nfft,nframe],order ='F')

i0 = 10
yi = yframe[:,i0]

srms = sr/1000
xms = np.arange(0,yi.size/srms,yi.size/srms/yi.size)
plt.plot(xms,yi)
plt.xlabel('Time in ms')
plt.ylabel('Sample Value')



class AudioFitFn(object):
    def __init__(self,yi,sr=44100,nterms=8):
        """
        A class for fitting 
        
        yi:  One frame of audio
        sr:  Sample rate (in Hz)
        nterms:  Number of harmonics used in the model (default=8)
        """
        self.yi = yi
        self.sr = sr
        self.nterms = nterms
                
    def feval(self,freq0):
        """
        Optimization function for audio fitting.  Given a fundamental frequency, freq0, the 
        method performs a least squares fit for the audio sample using the model:
        
        yhati[k] = c + \sum_{j=0}^{nterms-1} a[j]*cos(2*np.pi*k*freq0*(j+1)/sr) 
                                          +  b[j]*sin(2*np.pi*k*freq0*(j+1)/sr)
        
        The coefficients beta = [c,a[0],...,a[nterms-1],b[0],...,b[nterms-1]] 
        are found by least squares.

        Returns:
        
        mse1:   The MSE of the best least square fit.
        mse1_grad:  The gradient of mse1 wrt to the parameter freq0
        """
        
        # TODO   
        t1 = np.zeros(self.nterms)
        for i in range(self.nterms):
            t1[i] = 2*np.pi*freq0*(i+1)/self.sr
        t2 = np.zeros([self.yi.size,self.nterms])
        for i in range(self.yi.size):
            t2[i] = t1
        yt = np.arange(0,self.yi.size)
        yt1 = np.reshape(yt,[yt.size,1])
        yt2 = yt1*t2
        ytc = np.cos(yt2)
        yts = np.sin(yt2)
        ytf = np.column_stack((ytc,yts))
        A = np.column_stack((np.ones(ytf.shape[0],),ytf))
        betahat = np.linalg.lstsq(A,self.yi)[0]
        yhati = A.dot(betahat)


        mse1 = np.mean((yhati-self.yi) ** 2)
        mse1_grad = 0
        return mse1, mse1_grad



audio_fn = AudioFitFn(yi)
nfreq0 = np.linspace(40,500,100)
nmse = []
for freq0 in nfreq0:
    msei, mse_gradi = audio_fn.feval(freq0)
    nmse.append(msei)
nmse = np.array(nmse)
plt.figure()
plt.plot(nfreq0,nmse)

mse1_min_index = np.argsort(nmse)[0]
freq0_min = nfreq0[mse1_min_index]
print ("The value of freq0 that acieves the minimum mse1 is %f"%freq0_min)
plt.figure()
plt.plot(xms,yi,'r')


class AudioFitFn(object):
    def __init__(self,yi,sr=44100,nterms=8):

        self.yi = yi
        self.sr = sr
        self.nterms = nterms
                
    def feval(self,freq0):
        
        t1 = np.zeros(self.nterms)
        for i in range(self.nterms):
            t1[i] = 2*np.pi*freq0*(i+1)/self.sr
        t2 = np.zeros([self.yi.size,self.nterms])
        for i in range(self.yi.size):
            t2[i] = t1
        yt = np.arange(0,self.yi.size)
        yt1 = np.reshape(yt,[yt.size,1])
        yt2 = yt1*t2
        ytc = np.cos(yt2)
        yts = np.sin(yt2)
        ytf = np.column_stack((ytc,yts))
        A = np.column_stack((np.ones(ytf.shape[0],),ytf))
        betahat = np.linalg.lstsq(A,self.yi)[0]
        yhati = A.dot(betahat)

        mse1 = np.mean((yhati-self.yi) ** 2)
        
        m1 = 2 * (self.yi-yhati)
        mo = np.zeros(self.nterms)
        for i in range(self.nterms):
            mo[i] = 2*np.pi*(i+1)/self.sr
        ytc2 = mo * ytc
        yts2 = -mo * yts
        ytf2 = np.column_stack((yts2,ytc2))
        ytff = yt1 * ytf2
        A2 = np.column_stack((np.zeros(ytff.shape[0],),ytff))
        m2 = A2.dot(betahat)
        mse1_grad = np.mean(-m1*m2)
        return mse1, mse1_grad

# TODO
# Take a random initial point
freq0_0 = 460*np.random.rand(1)+40
print(freq0_0)
         # Perturb the point
step = 1e-6
freq0_1 = freq0_0 + step*freq0_0
audio_fn = AudioFitFn(yi)

# Measure the function and gradient at w0 and w1
f0, fgrad0 = audio_fn.feval(freq0_0)
f1, fgrad1 = audio_fn.feval(freq0_1)
# Predict the amount the function should have changed based on the gradient
df_est = fgrad0*(freq0_1-freq0_0)
# Print the two values to see if they are close
print("Actual f1-f0 = %12.4e" % (f1-f0))
print("Predicted f1-f0 = %12.4e" % df_est)


def grad_opt_adapt(feval, winit, nit=1000, lr_init=1e-3):
    """
    Gradient descent optimization with adaptive step size
    
    feval:  A function that returns f, fgrad, the objective
            function and its gradient
    winit:  Initial estimate
    nit:    Number of iterations
    lr:     Initial learning rate
    
    Returns:
    w:   Final estimate for the optimal
    f0:  Function at the optimal
    """
    
    # Set initial point
    w0 = winit
    f0, fgrad0 = feval(w0)
    lr = lr_init
    
    # Create history dictionary for tracking progress per iteration.
    # This isn't necessary if you just want the final answer, but it 
    # is useful for debugging
    hist = {'lr': [], 'w': [], 'f': []}

    for it in range(nit):

        # Take a gradient step
        w1 = w0 - lr*fgrad0

        # Evaluate the test point by computing the objective function, f1,
        # at the test point and the predicted decrease, df_est
        f1, fgrad1 = feval(w1)
        df_est = fgrad0*(w1-w0)
        
        # Check if test point passes the Armijo rule
        alpha = 0.5
        if (f1-f0 < alpha*df_est) and (f1 < f0):
            # If descent is sufficient, accept the point and increase the
            # learning rate
            lr = lr*2
            f0 = f1
            fgrad0 = fgrad1
            w0 = w1
        else:
            # Otherwise, decrease the learning rate
            lr = lr/2            
            
        # Save history
        hist['f'].append(f0)
        hist['lr'].append(lr)
        hist['w'].append(w0)

    # Convert to numpy arrays
    for elem in ('f', 'lr', 'w'):
        hist[elem] = np.array(hist[elem])
    return w0, f0, hist

freq0 = 130
feval = audio_fn.feval
freqFinal, mseFinal, hist = grad_opt_adapt(feval, freq0)

print ("The final frequency estimate is %f"%freqFinal)

midi_num = 12*np.log2(freqFinal/440) + 69
print ("The midi number of the estimated frequency is %f"%midi_num)


plt.figure()
plt.loglog(hist['f'],'r')
plt.xlabel('Iteration')
plt.ylabel('MSE')

freq0 = 200
freqFinal, mseFinal, hist = grad_opt_adapt(feval, freq0)
plt.loglog(hist['f'],'g')



























