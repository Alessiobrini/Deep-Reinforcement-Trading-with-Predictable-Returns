# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:49:11 2021

@author: alessiobrini
"""

import matplotlib.pyplot as plt
import numpy as np
import pdb
from common import set_size

import seaborn as sns

sns.set_style("darkgrid")

# parameters
phi = 0.025
omega = 0.1
sigma = 0.1
gamma = 0.01
lamb= 0.001
rho = 0.0
b = 0.004

points = 100
label = 'Varying parameter' #'$\lambda$'
# phi = x =  np.linspace(0.00001,0.0005,points) # speed of mean reversion
omega = x =  np.linspace(0.0001,0.05,points) # factor volatility
# sigma = x =  np.linspace(0.001,0.01,points) # return volatility
# gamma = x = np.linspace(0.0001,0.001,points) # risk aversion
# lamb = x =  np.linspace(0.001,0.1,points) # cost multiplier
# rho = x =  np.linspace(0.00001,0.999,points) # discount factor
# b = x = np.linspace(0.00001,0.05,points) # factor loading

# Trading rate
num1 = gamma * (1 - rho) + lamb * rho
num2 = np.sqrt(
    num1 ** 2 + 4 * gamma * lamb * (1 - rho) ** 2
)
den = 2 * (1 - rho)
a = (-num1 + num2) / den
trate = a/lamb
a_gamma = trate * lamb / gamma

# Vf parameters
rho_bar = 1 - rho
lamb_bar = lamb/rho_bar
Lamb_bar = sigma**2 / lamb_bar
axx = ( rho_bar*gamma *Lamb_bar**(0.25)*sigma**2 + 0.25*(rho*Lamb_bar**2 + 2*rho*gamma * Lamb_bar**0.25*sigma**2 + \
                                                          gamma**2 * Lamb_bar**(-3/4)*sigma**2)**(0.5) - 0.5*(rho*Lamb_bar + gamma*sigma**2)) 
axf = rho_bar*( ((1 - axx/Lamb_bar)* b) /(1 - rho_bar* (1 - phi) * (1 - axx/Lamb_bar)))
aff = (rho_bar/ (1 - rho_bar* (1-phi)**2)) * ((b + axf*(1-phi))**2 / (gamma * sigma**2 + Lamb_bar + axx))

# Grouped constants
c1 = 1-trate
c2 = trate * (1/(gamma * sigma**2)) * (b/(1+phi*a_gamma))
cxf = (c2 * (1-phi)) /(1-c1 * (1-phi))

p = (c1**2 * sigma**2 * ((c2**2+2*c1*c2*cxf)/c1**2) + c2**2*sigma**2 + ((2*c1*c2*(1-phi)**2)/(1-c1*(1-phi)**3)) * \
(3*c1*(1-phi)*cxf*sigma**2 + 2*c1*c2*sigma**2*cxf)) / (1-c1**2*(1-phi)**2)
q = (3*c2**2*(1-phi) + c2*(1-phi)**3) / (1-c1**2*(1-phi)**2) 

# functions
f_func = omega**2 / (1- (1-phi)**2)
h_func = ((c2**2 + 2*c1*c2*cxf)/(c1**2)) *  f_func
V_func = (0.5*axx**2 * (c2**2 + 2*c1*c2*cxf**2)/(c1**2) + axf**2* (q- cxf**2) + 0.5* aff**2) * f_func**2 + axf**2 * p * f_func



fig,(ax1,ax2,ax3) = plt.subplots(3,1, figsize=set_size(1000))
fig.subplots_adjust(hspace=0.4)

if isinstance(f_func,float): f_func = np.repeat(f_func,x.shape[0])
ax1.plot(x,f_func)
ax1.set_ylabel('$\mathrm{Var}[f_{t}]$')

if isinstance(h_func,float): h_func = np.repeat(h_func,x.shape[0])
ax2.plot(x,h_func)
ax2.set_ylabel('$\mathrm{Var}[x_{t}]$')


ax3.plot(x,V_func)
ax3.set_ylabel('$\mathrm{Var}[V(x_{t-1},f_{t})]$')
ax3.set_xlabel(label)