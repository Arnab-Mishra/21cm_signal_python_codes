#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2,ifft, fftfreq, fftshift,irfft,rfft,irfft2
import random
from beam import Beam
from scipy.special import j1
from cosmopara import E,dTb,drdnu,rc,radius,P_I,I
from scipy.integrate import simps


# In[2]:


def heaviside(x):
    if x < 0:
        return 0
    else:
        return 1


# In[3]:


def Signal(r):
    N = 512
    nuc = 150e6
    dnu = 62.5e3
    nu0=nuc-128*dnu
    BW = dnu * N // 2  # Bandwidth
    nu_array = np.arange(nu0, nu0 + BW, dnu)
    z_array = (1420e6 / nu_array) - 1
    zc=z_array[N//4]
    om=.27 # matter density paramete
    I0=I(zc,om)
    rad=radius(r,zc,om)
    A=I0*(np.zeros((N//2,N,N))+1) 

    x0,y0,z0=N//2,N//2,N//4

    for z in range(z0-rad, z0+rad+1):
        for x in range(x0-rad, x0+rad+1):
            for y in range(y0-rad, y0+rad+1):

                S = rad - ((x0-x)**2 + (y0-y)**2 + (z0-z)**2)**0.5
                if S>=0: A[z,x,y] = 0
    return A;


# In[5]:

def SignalFFT(r):
    om=.27 # matter density parameter
    dnu1=0.0625
    N = 512
    nuc = 150e6
    dnu = 62.5e3
    nu0=nuc-128*dnu
    BW = dnu * N // 2  # Bandwidth
    nu_array = np.arange(nu0, nu0 + BW, dnu)
    z_array = (1420e6 / nu_array) - 1
    zc=z_array[N//4]
    om=.27 # matter density paramete
    
    r1=rc(zc,om)
    l=drdnu(zc,om)*dnu1 #resolution in MPc
    L=l/r1 #resolution in radian
    A=Signal(r)
    SFT=np.zeros((N//2,N,N), dtype=complex)
    for i in range(N//2):
        SFT[i]=L**2*(fftshift(fft2(A[i])))
    SFT[:,256,256]=0
    SFT1=np.zeros((N//2,N,N), dtype=complex)
    for i in range(N//2):
        for j in range(N):
            for k in range(N):
                
                SFT1[i,j,k] = (-1)**(j + k) * SFT[i,j,k]
    return SFT1


# In[6]:



# In[7]:


def Baseline(r):
    N = 512
    nuc = 150e6
    dnu = 62.5e3
    nu0=nuc-128*dnu
    BW = dnu * N // 2  # Bandwidth
    nu_array = np.arange(nu0, nu0 + BW, dnu)
    z_array = (1420e6 / nu_array) - 1
    zc=z_array[N//4]
    om=.27 # matter density parameter
    dnu1=0.0625
    r1=rc(zc,om)
    l=drdnu(zc,om)*dnu1 #resolution in MPc
    L=l/r1 #resolution in radian
    N=512 # grid points
    A=Signal(r)
    
    SFT=np.zeros((N//2,N,N), dtype=complex)
    
    SFT=L**2*(fftshift(fft2(A[1])))
    SFT[256,256]=0
    freq_x=fftshift(fftfreq(SFT[128,:].size,L))
    freq_y=fftshift(fftfreq(SFT[:,128].size,L))
    X, Y = np.meshgrid(freq_x, freq_y)
    
    # Calculate the Baselines
    U = np.sqrt(X**2+Y**2)
    return U;

# In[19]:

def ch_no(r):
    om = 0.27
    N = 512
    nuc = 150e6
    dnu = 62.5e3
    zc= (1420e6 / nuc) - 1
    r1 = rc(zc, om)
    r_ = drdnu(zc, om)
    dnub =  (10**6)*r / r_
    nu_array = np.arange(nuc-dnub, nuc + dnub, dnu)
    nn=len(nu_array)
    return nn

# In[20]:

def F(r):
    om = 0.27
    N = 512
    nuc = 150e6
    zc=(1420e6/nuc)-1
    rnuc = rc(zc, om)
    r_ = drdnu(zc, om)
    dnub =  (10**6)*r / r_
    I0 = I(zc, om)
    U = Baseline(r)
    dnu=62.5e3
    BW=dnu*N//2
    dnu_array=np.arange(-BW//2,BW//2,dnu)
    Filter = np.zeros((N // 2, N, N))
    for i,dnu1 in enumerate(dnu_array):
        if 1 - (dnu1 / dnub)**2 >= 10**-5:
            R_nu = r * (1 - ((dnu1) / dnub)**2)**0.5
        else:
            R_nu = 0

        # angular radius circular disk
        ThR = R_nu / rnuc
        y=1-np.abs(dnu1)/dnub
        x = 2 * np.pi * ThR * U
        Filter[i] = -np.pi * I0 * (ThR ** 2) * (np.where(x == 0, 1, 2 * j1(x) / x)) * heaviside(y)

    return Filter

# In[21]:

def FI(r):
    om = 0.27
    N = 512
    nuc = 150e6
    zc=(1420e6/nuc)-1
    rnuc = rc(zc, om)
    r_ = drdnu(zc, om)
    dnub =  (10**6)*r / r_
    I0 = I(zc, om)
    U_array = Baseline(r)
    Dnu=62.5e3
    BW=Dnu*N//2
    dnu_array=np.arange(-BW//2,BW//2,Dnu)
    FilterI = np.zeros((N // 2, N, N))
    B=4*dnub
    def Int(U):
        def Visb(dnu):
            if 1 - (dnu/ dnub)**2 >= 10**-5:
                R_nu = r * (1 - ((dnu) / dnub)**2)**0.5
            else:
                R_nu = 0
             # angular radius circular disk
            ThR = R_nu / rnuc
            y=1-np.abs(dnu)/dnub
            x = 2 * np.pi * ThR * U
            V= -np.pi * I0 * (ThR ** 2) * (np.where(x == 0, 1, 2 * j1(x) / x)) * heaviside(y)
            return V
        Visb_vect=np.vectorize(Visb)
        Dnu_array=np.linspace(-B/2,B/2,6)
        Vis=Visb_vect(Dnu_array)
        Integration=simps(Vis,Dnu_array)
        return Integration
    
    I_vec=np.vectorize(Int)
    In=I_vec(U_array)
    
    for i,dnu1 in enumerate(dnu_array):
        if 1 - (dnu1 / dnub)**2 >= 10**-5:
            R_nu = r * (1 - ((dnu1) / dnub)**2)**0.5
        else:
            R_nu = 0

        # angular radius circular disk
        ThR = R_nu / rnuc
        x = 2 * np.pi * ThR * U_array
        y=1-np.abs(dnu1)/dnub
        w=1-2*np.abs(dnu1)/B
        FilterI[i] = -np.pi*I0*(ThR**2)*(np.where(x == 0, 1, 2 * j1(x) / x))*heaviside(y) - (In*heaviside(w)/B)

    return FilterI
    
    
