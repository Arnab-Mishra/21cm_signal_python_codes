#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2,ifft, fftfreq, fftshift,irfft,rfft,irfft2
import random
from scipy.special import j1
from beam import Beam
from cosmopara import E,dTb,drdnu,rc,radius,P_I,I
from Signal import Signal,Baseline,SignalFFT,ch_no,F,FI
from Noise import noise_rms1
from scipy.integrate import simps


# In[2]:

N=512 # grid points
n1=211
n2=300
n=n2-n1


# In[3]:


R=np.arange(10,51,10)
Estimator=np.zeros(len(R))
Estimator1=np.zeros(len(R))
j=-1
j1=-1
for r in R:
    ch=ch_no(r)
    
    Signal_FT=SignalFFT(r)[128-ch//2:128+ch//2,n1:n2,n1:n2]
    Visb=np.zeros((ch,n,n), dtype=complex)
    Visb=(Signal_FT)
    
    #for matched filter
    filter0=F(r)[128-ch//2:128+ch//2,n1:n2,n1:n2]

    E=Visb*filter0

    E_sum=np.sum(E)/(ch*n*n)
    j=j+1
    Estimator[j]=np.abs(E_sum)
    
    
    #for FilterI
    filter1=FI(r)[128-ch//2:128+ch//2,n1:n2,n1:n2]

    E1=Visb*filter1
    E_sum1=np.sum(E1)/(ch*n*n)
    j1=j1+1
    Estimator1[j1]=np.abs(E_sum1)


# In[4]:


plt.loglog(R,Estimator*10**6)
plt.loglog(R,Estimator1*10**6)
plt.legend(['Match Filter','FilterI'])
plt.ylabel(r'$<E>(mJy^2)$')
plt.xlabel(r'Bubble Size')
plt.title("matched_Filter and FilterI")


# In[5]:


Err=np.zeros(len(R))
for i in range(len(R)):
    
    Err[i]=(Estimator[i]-Estimator1[i])/Estimator[i]
plt.plot(R,Err)
plt.ylabel(r'$(E_{matched}-E_I)/E_{matched}$')
plt.xlabel(r'Bubble Size')
plt.title("Differences between the two estimators for the filters")


# In[6]:


Estimator1*10**6


# In[6]:



R1=np.arange(20,51,15)
dE=np.zeros(len(R1))
k=-1
for r1 in R1:
    ch=ch_no(r1)
    B=ch*0.0625 #Bandwidth
    tobs1=100 #hours
    sigma=noise_rms1(B,tobs1)
    Filter=FI(r1)[128-ch//2:128+ch//2,n1:n2,n1:n2]

    DE=(sigma**2)*np.abs(Filter)**2
    dE_sum=np.sum(DE)/(ch*n*n)
    k=k+1
    dE[k]=np.abs(dE_sum)


# In[7]:


3*np.sqrt(dE)*10**6


# In[8]:


(Estimator-3*np.sqrt(dE))*10**6


# In[9]:



R2=np.arange(20,51,15)
dE1=np.zeros(len(R2))
k1=-1
for r2 in R2:
    ch=ch_no(r2)
    B=ch*0.0625 #Bandwidth
    tobs1=10000 #hours
    sigma1=noise_rms1(B,tobs1)
    Filter1=FI(r2)[128-ch//2:128+ch//2,n1:n2,n1:n2]

    DE1=(sigma1**2)*np.abs(Filter1)**2
    dE_sum1=np.sum(DE1)/(ch*n*n)
    k1=k1+1
    dE1[k1]=np.abs(dE_sum1)


# In[10]:


plt.loglog(R,Estimator*10**6)
plt.loglog(R,3*np.sqrt(dE)*10**6)
plt.loglog(R,3*np.sqrt(dE1)*10**6)
plt.legend(['signal','noise-100 hours','noise-$10^4$ hours'])
plt.ylabel(r'$<E>(mJy^2)$')
plt.xlabel(r'Bubble Size')
plt.title("Estimator considering both signal and noise")


# In[11]:


(Estimator-3*np.sqrt(dE1))*10**6


# In[ ]:




