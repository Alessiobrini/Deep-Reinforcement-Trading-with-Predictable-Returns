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

plot_2d = False
labelsize=10

# parameters
variable_dt = True
dt=0.5 #1.0
phi = 0.025
omega = 0.1
sigma = 0.1
gamma = 0.01
lamb= 0.001
rho = 0.0
b = 0.004

points = 1000

if plot_2d:
    # if variable_dt:
    #     dt, labelx = x, labelx =  np.linspace(0.1,1.0,points), '$dt$' # speed of mean reversion
    #     phi = phi * dt
    # else:    
    #     phi, labelx = x, labelx =  np.linspace(0.00001,0.05,points)* dt, '$\phi$' # speed of mean reversion
    # omega, labelx = x, labelx =  np.linspace(0.001,0.02,points), '$\omega$'  # factor volatility
    # sigma, labelx = x, labelx =  np.linspace(0.001,0.01,points), '$\sigma$' # return volatility
    # gamma, labelx = x, labelx = np.linspace(0.0001,0.001,points), '$\gamma$' # risk aversion
    # lamb, labelx = x, labelx =  np.linspace(0.001,0.1,points), '$\lambda$' # cost multiplier
    # rho, labelx = x, labelx =  np.linspace(0.00001,0.999,points), '$\rho$' # discount factor
    b, labelx = x, labelx = np.linspace(0.005,0.1,points), '$b$'  # factor loading
else:
    if variable_dt:
        dt, labelx = x, labelx =  np.linspace(0.1,1.0,points), '$dt$' # speed of mean reversion
        phi = phi * dt
    else:    
        phi, labelx = x, labelx =  np.linspace(0.00001,0.0005,points)* dt, '$\phi$' # sp
    omega, labely = y, labely =  np.linspace(0.01,0.2,points), '$\omega$'  # factor volatility
    # sigma, labely = y, labely =  np.linspace(0.005,0.05,points), '$\sigma$' # return volatility
    # gamma, labelx = x, labelx = np.linspace(0.0001,0.001,points), '$\gamma$' # risk aversion
    # lamb, labelx = x, labelx =  np.linspace(0.001,0.1,points), '$\lambda$' # cost multiplier
    # rho, labelx = x, labelx =  np.linspace(0.00001,0.999,points), '$\rho$' # discount factor
    # b, labely = y, labely = np.linspace(0.00001,0.05,points), '$b$'  # factor loading

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

p = (c1**2 * sigma**2 * ((c2**2+2*c1*c2*cxf)/c1**2) + c2**2*sigma**2 + ((6*c1*c2*(1-phi)**2)/(1-c1*(1-phi)**3)) * \
(c1*(1-phi)*cxf*sigma**2 + c2*(1-phi)*sigma**2 +  2*c1*c2*sigma**2*cxf)) / (1-c1**2*(1-phi)**2)
q = (3*c2**2*(1-phi) + (6*c1*c2*(1-phi)**2 * c2*(1-phi)**3)/(1-c1*(1-phi)**3) ) / (1-c1**2*(1-phi)**2) 

# functions
f_func = omega**2 / (1- (1-phi*dt)**2)
h_func = ((c2**2 + 2*c1*c2*cxf)/(c1**2)) *  f_func
V_func = (0.5*axx**2 * (c2**2 + 2*c1*c2*cxf**2)/(c1**2) + axf**2* (q- cxf**2) + 0.5* aff**2) * f_func**2 + axf**2 * p * f_func
grosspnl_func_var = (c1**2*b**2*p + c1**2 * ((c2**2 + 2*c1*c2*cxf)/c1**2)* sigma**2 + c2**2 + (6*c1*c2*b**2)/(1-c1*(1-phi)**3) *(c1*(1-phi)*cxf*sigma**2 + c2*(1-phi)*sigma**2) + \
                 2*c1*c2*cxf*sigma**2 - c1*b*cxf + c2*b) * f_func \
    + (c1**2 * b**2 * (q-cxf**2) + 3*c2**2*b**2 + (6*c1*c2*b**2)/(1-c1*(1-phi)**3) * c2*(1-phi)**2) * f_func**2
    

grosspnl_func_expval = (c1*b*cxf + c2*b) * f_func

if plot_2d:

    fig,(ax_row1,ax_row2,ax_row3) = plt.subplots(3,2, figsize=set_size(1200))
    fig.subplots_adjust(wspace=0.5)
    
    if isinstance(f_func,float): f_func = np.repeat(f_func,x.shape[0])
    ax_row1[0].plot(x,f_func)
    ax_row1[0].set_ylabel('$\mathrm{Var}[f_{t}]$')
    
    if isinstance(h_func,float): h_func = np.repeat(h_func,x.shape[0])
    ax_row1[1].plot(x,h_func)
    ax_row1[1].set_ylabel('$\mathrm{Var}[x_{t}]$')
    
    
    ax_row2[0].plot(x,V_func)
    ax_row2[0].set_ylabel('$\mathrm{Var}[V(x_{t-1},f_{t})]$')

    
    ax_row2[1].plot(x,grosspnl_func_var)
    ax_row2[1].set_ylabel('$\mathrm{Var}[x_{t-1}r_{t}]$')

    
    ax_row3[0].plot(x,grosspnl_func_expval)
    ax_row3[0].set_ylabel('$\mathrm{E}[x_{t-1}r_{t}]$')
    ax_row3[0].set_xlabel(labelx)
    
else:

    fig = plt.figure(figsize=set_size(1000))
    # fig.subplots_adjust(wspace=0.5)

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')



    if isinstance(f_func,float): f_func = np.repeat(f_func,x.shape[0])
    ax1.plot3D(x,y,f_func)
    ax1.set_title('$\mathrm{Var}[f_{t}]$')
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely)
    ax1.xaxis.set_rotate_label(False) 
    ax1.yaxis.set_rotate_label(False) 
    ax1.xaxis.set_tick_params(labelsize=labelsize)
    ax1.yaxis.set_tick_params(labelsize=labelsize)
    ax1.zaxis.set_tick_params(labelsize=labelsize)
    
    if isinstance(h_func,float): h_func = np.repeat(h_func,x.shape[0])
    ax2.plot3D(x,y,h_func)
    ax2.set_title('$\mathrm{Var}[x_{t}]$')
    ax2.set_xlabel(labelx)
    ax2.set_ylabel(labely)
    ax2.xaxis.set_rotate_label(False) 
    ax2.yaxis.set_rotate_label(False) 
    ax2.xaxis.set_tick_params(labelsize=labelsize)
    ax2.yaxis.set_tick_params(labelsize=labelsize)
    ax2.zaxis.set_tick_params(labelsize=labelsize)
    
    
    ax3.plot3D(x,y,V_func)
    ax3.set_title('$\mathrm{Var}[V(x_{t-1},f_{t})]$')
    ax3.set_xlabel(labelx)
    ax3.set_ylabel(labely)
    ax3.xaxis.set_rotate_label(False) 
    ax3.yaxis.set_rotate_label(False) 
    ax3.xaxis.set_tick_params(labelsize=labelsize)
    ax3.yaxis.set_tick_params(labelsize=labelsize)
    ax3.zaxis.set_tick_params(labelsize=labelsize)
    
    ax4.plot3D(x,y,grosspnl_func_var)
    ax4.set_title('$\mathrm{Var}[x_{t-1}r_{t}]$')
    ax4.set_xlabel(labelx)
    ax4.set_ylabel(labely)
    ax4.xaxis.set_rotate_label(False) 
    ax4.yaxis.set_rotate_label(False) 
    ax4.xaxis.set_tick_params(labelsize=labelsize)
    ax4.yaxis.set_tick_params(labelsize=labelsize)
    ax4.zaxis.set_tick_params(labelsize=labelsize)
    
    ax5.plot3D(x,y,grosspnl_func_expval)
    ax5.set_title('$\mathrm{E}[x_{t-1}r_{t}]$')
    ax5.set_xlabel(labelx)
    ax5.set_ylabel(labely)
    ax5.xaxis.set_rotate_label(False) 
    ax5.yaxis.set_rotate_label(False) 
    ax5.xaxis.set_tick_params(labelsize=labelsize)
    ax5.yaxis.set_tick_params(labelsize=labelsize)
    ax5.zaxis.set_tick_params(labelsize=labelsize)

    ax6.plot3D(y,grosspnl_func_var,grosspnl_func_expval)
    ax6.set_title('$\mathrm{E}[x_{t-1}r_{t}]$')
    ax6.set_xlabel(labely)
    ax6.set_ylabel('$\mathrm{Var}[x_{t-1}r_{t}]$')
    ax6.set_zlabel('$\mathrm{E}[x_{t-1}r_{t}]$')
    ax6.xaxis.set_rotate_label(False) 
    ax6.yaxis.set_rotate_label(False) 
    ax6.xaxis.set_tick_params(labelsize=labelsize)
    ax6.yaxis.set_tick_params(labelsize=labelsize)
    ax6.zaxis.set_tick_params(labelsize=labelsize)
