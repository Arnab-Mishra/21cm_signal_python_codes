#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
#definig beam function

def Beam(pos,freq):
    D=45
    d=1
    c=3*10**8
    nubyc=freq/c
    D=D*nubyc
    d=d*nubyc
    arg=np.pi*pos*D
    if pos==0:
        be=1
    else:
        be=2*(j1(arg)/arg)
        be=be*be
    return be


# In[ ]:




