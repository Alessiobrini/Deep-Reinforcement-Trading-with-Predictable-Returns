# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:32:29 2020

@author: aless
"""
from typing import Tuple, Union
import numpy as np
from tqdm import tqdm
import pdb
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn 
seaborn.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['savefig.dpi'] = 90
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

def ReturnSampler(N_train : int,
                  sigmaf : Union[float or list or np.ndarray],
                  f0 : Union[float or list or np.ndarray],
                  f_param: Union[float or list or np.ndarray],
                  sigma: Union[float or list or np.ndarray],
                  plot_inputs: int,
                  HalfLife: Union[int or list or np.ndarray],
                  seed: int = None,
                  adftest: bool = False) -> Tuple[Union[list or np.ndarray],
                                                          Union[list or np.ndarray],
                                                          Union[list or np.ndarray]]:

    # Set seed to make the out-of-sample experiment reproducible
    np.random.seed(seed)
    
    # use samplesize +2 because when iterating the algorithm is necessary to 
    # have one observation more (the last space representation) and because
    # we want be able to plot insample operation every tousand observation.
    # Therefore we don't want the index ending at 999 instead of 1000
    
    # Generate stochastic factor component and compute speed of mean reversion
    # simulate the single factor according to OU process
    # select proper speed of mean reversion and initialization point
    # it is faster to increase the size of a python list than a numpy array
    # therefore we convert later the list
    eps = np.random.randn(N_train + 2) 
    lambdas = np.around(np.log(2)/HalfLife,4)
    f = []
    
    # possibility of triple noise
    for i in tqdm(iterable=range(N_train + 2), desc='Simulating Factors'):
        # multiply makes the hadamard (componentwise) product
        # if we want to add different volatility for different factors we could
        # add multiply also the the second part of the equation
        f1 = np.multiply((1 - lambdas),f0) + np.multiply(sigmaf,eps[i])
        f.append(f1)
        f0 = f1

    factors = np.vstack(f)
    
    np.random.seed(seed)
    
    u = np.random.randn(N_train + 2)
    # now we add noise to the equation of return by default, while in the previous
    # implementation we were using a boolean
    # single noise
    realret = np.sum(f_param * factors, axis=1) + sigma * u
    f_speed = lambdas
                

    
    # plots for factor, returns and prices
    if plot_inputs:
        print(str(len(np.atleast_2d(f_speed))), 'factor(s) simulated')
        print('################################################################')
        print('max realret ' + str(max(realret)))
        print('min realret ' + str(min(realret)))
        print('################################################################')
        fig1 = plt.figure()
        fig2 = plt.figure()

        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)
            
        ax1.plot(factors)
        ax1.legend(['5D','1Y','5Y'])
        ax1.set_title('Factors')
        ax2.plot(realret)
        plt.legend(['CapReturns','Returns'])
        ax2.set_title('Returns')

        fig1.show()
        fig2.show()
    if adftest:
        test=adfuller(realret)
        # print('Test ADF for generated return series')
        # print("Test Statistic: " + str(test[0]))
        # print("P-value: " + str(test[1]))
        # print("Used lag: " + str(test[2]))
        # print("Number of observations: " + str(test[3]))
        # print("Critical Values: " + str(test[4]))
        # print("AIC: " + str(test[5]))
        return realret.astype(np.float32),factors.astype(np.float32), f_speed, test
    
    
    return realret.astype(np.float32),factors.astype(np.float32), f_speed