#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import integrate


# In[2]:


#Defining Ez

def E(z,om):
    ol=1-om
    h=0.7
    ombh =.022 
    cH_0=3000/h  #in Mpc unit
    Ez = (om*(1+z)**3+ol)**(.5)
    return Ez


# In[3]:


#defining dTb

def dTb(z,om):
    h=0.7
    Ez=E(z,om)
    ombh=0.022
    dT_b=4*(ombh/0.02)*(0.7/h)*(1/Ez)*(1+z)**2
    return dT_b


# In[4]:


#defining background HI intensity

def I(z,om):
    h=0.7
    Ez=E(z,om)
    ombh=0.022
    I0=(2.5*10**2)*(ombh/0.02)*(0.7/h)*(1/Ez) #jy/sr
    return I0


# In[5]:


#defining drdnu

def drdnu(z,om):
    nu=1420/(1+z)
    h=0.7
    cH_0=3000/h
    Ez=E(z,om)
    drdnu=1420*cH_0*(nu**-2)/Ez
    return drdnu


# In[6]:


#defining rc

def rc(z,om):
    z1=np.arange(0, z,.00001)
    ol=1-om
    Ez1 = (om*(1+z1)**3+ol)**(.5)
    h=0.7
    cH_0=3000/h
    R =cH_0/Ez1
    r=integrate.simps(R,z1) # in Mpc unit
    return r 
    


# In[7]:


# defining grid  radius
#here r is comoving radius

def radius(r,z,om):
    dnu=0.0625 #MHz unit
    L=drdnu(z,om)*dnu
    radius =int(np.floor(r/L)) 
    return radius
    


# In[8]:


#Defining power spectrum

def P_I(nu,U):
    A_150=513*10**-6 #K^2 unit
    betav=2.34
    nu0=150*10**6 #Hz unit
    alpha=2.8
    k_B=1.38*10**3 # in Jy unit
    c=3*10**8
    #pu=A_150*pow(1000./(2.*np.pi*U),betav)*pow(nu0/nu,2.0*alpha)
    pu=A_150*pow(1000./(2.*np.pi*U),betav)*pow(2.*k_B*nu*nu/(c*c),2.)*pow(nu0/nu,2.0*alpha)
    return pu


# In[ ]:





# In[ ]:




