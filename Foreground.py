#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2,ifft, fftfreq, fftshift,irfft,rfft,irfft2
import random
from beam import Beam
from cosmopara import E,dTb,drdnu,rc,radius,P_I


# In[2]:


def Foreground(z):
    om=.27 # matter density parameter
    r=rc(z,om)
    dnu1=0.0625
    l=drdnu(z,om)*dnu1 #resolution in MPc
    L=l/r
    N=512 # grid points
    ydim =N//2+1
    xdim=N
    #fac=N/(L*np.sqrt(2.))#factor for temp/strd fluctuations
    fac=N*L/np.sqrt(2.) #for intensity
    length=L*N # total length in rad
    nu0=150*10**6
    dnu=62.5*10**3
    BW=dnu*N//2 #Bandwidth
    #creating an array for simulating in multichannel
    nu_array=np.arange(nu0,nu0+BW,dnu)
    sigma=1
    mean=0
    
    #Generating the Foreground signal

    #allocating memory of the array
    in_array = np.zeros((N//2,N,(N//2+1)), dtype=complex)
    np.random.seed(850)


    ##Filling Fourier Components along axis (j-0 and j=N/2)
    k=-1
    for nu in nu_array:    
        k+=1
        for j in range(0, N//2+1 , N//2):
            for i in range(1, N//2):
                # along + x
                u = np.sqrt(1.0 * (i * i + j * j)) / length
                amp = fac * np.sqrt(P_I(nu,u))            
                in_array.real[k,i,j] = pow(-1, i + j) * amp * np.random.normal(mean, sigma)
                in_array.imag[k,i,j] = pow(-1, i + j) * amp * np.random.normal(mean, sigma)
                 #-x axis
                index1=(N-i)*ydim+j
                in_array.real[k,N-i,j]=in_array.real[k,i,j]
                in_array.imag[k,N-i,j]=-in_array.imag[k,i,j]

                

    # upper half plane excluding x axis
    k1=-1
    for nu in nu_array:    
        k1+=1
        for i in range(xdim):
            for j in range(1, N//2):
                ia = N - i if i > N//2 else i
                u = np.sqrt(1.0 * (ia * ia + j * j)) / length
                amp = fac * np.sqrt(P_I(nu,u))
                in_array.real[k1,i,j] = pow(-1, i + j) * amp * np.random.normal(mean, sigma)
                in_array.imag[k1,i,j] = pow(-1, i + j) * amp * np.random.normal(mean, sigma)

    #remaining 4 points 
    k2=-1
    for nu in nu_array:    
        k2+=1
        for i in range(2):
            for j in range(2):
                if i + j == 0:
                    in_array.real[k2,i,j] = 0.0
                    in_array.imag[k2,i,j] = 0.0
                else:
                    u = (N//2) * np.sqrt(1.0 * (i * i + j * j)) / length
                    amp = fac * np.sqrt(P_I(nu,u))
                    index = i * (N//2) * ydim + j * (N//2)
                    in_array.real[k2,i*(N//2),j*(N//2)] = pow(-1, (i * N//2 + j * N//2)) * amp * np.random.normal(mean, sigma)
                    in_array.imag[k2,i*(N//2),j*(N//2)] = 0.0
                    
                    
    #imaginary to real fourier transform of the in_array

    out=np.zeros((N//2,N,N))
    for i in range(N//2):
        out[i]=(irfft2(in_array[i]))/L**2
    return out;






