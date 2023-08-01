#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2,ifft, fftfreq, fftshift,irfft,rfft,irfft2
import random



# In[5]:


def noise_rms1(B,tobs):
    Cx=1.03 #Jy
    N=30
    Nb=N*(N-1)/2
    rms=Cx/np.sqrt(2*Nb*B*tobs*3600) #tobs in hours and B is in MHz 
    return rms


# In[8]:


noise_rms1(6,100)


# In[ ]:




