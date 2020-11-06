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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_arch
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
                  rng: int = None,
                  offset: int = 2,
                  adftest: bool = False,
                  uncorrelated: bool = False,
                  seed_test: int = None,
                  hetersk : bool = False,
                  alpha_h : float = None,
                  beta_h : float = None) -> Tuple[Union[list or np.ndarray],
                                                          Union[list or np.ndarray],
                                                          Union[list or np.ndarray]]:

    
    # Set seed to make the out-of-sample experiment reproducible
    #np.random.seed(seed)
    if seed_test is not None:
        rng = np.random.RandomState(seed_test*2)
    
    # use samplesize +2 because when iterating the algorithm is necessary to 
    # have one observation more (the last space representation) and because
    # we want be able to plot insample operation every tousand observation.
    # Therefore we don't want the index ending at 999 instead of 1000
    
    # Generate stochastic factor component and compute speed of mean reversion
    # simulate the single factor according to OU process
    # select proper speed of mean reversion and initialization point
    # it is faster to increase the size of a python list than a numpy array
    # therefore we convert later the list
    if uncorrelated:
        eps = rng.randn(N_train + offset, len(HalfLife)) 
    else:
        eps = rng.randn(N_train + offset)

    lambdas = np.around(np.log(2)/HalfLife,4)
    f = []
    
    # possibility of triple noise
    for i in tqdm(iterable=range(N_train + offset), desc='Simulating Factors'):
        # multiply makes the hadamard (componentwise) product
        # if we want to add different volatility for different factors we could
        # add multiply also the the second part of the equation
        f1 = np.multiply((1 - lambdas),f0) + np.multiply(sigmaf,eps[i])
        f.append(f1)
        f0 = f1

    factors = np.vstack(f)

    if hetersk:
        u = rng.randn(N_train + offset)
        # now we add noise to the equation of return by default, while in the previous
        # implementation we were using a boolean
        # single noise
        
        noises = []
        n0 = sigma * float(rng.randn(1))
        for i in tqdm(iterable=range(N_train + offset), desc='Simulating Returns'):
            sigma_sq = (alpha_h * n0**2) + (beta_h *  sigma**2)
            news = np.sqrt(sigma_sq) * u[i]
            n0=news
            sigma = np.sqrt(sigma_sq)
            noises.append(news)
        pdb.set_trace()
        realret = np.sum(f_param * factors, axis=1) + noises
        f_speed = lambdas
    else:
        #np.random.seed(seed)
        u = rng.randn(N_train + offset)
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


def create_lstm_tensor(X, look_back=5):
        
    dataX= []
    for i in tqdm(iterable=range(len(X)-look_back+1), desc='Creating tensors for LSTM'):
        a = X[i:(i+look_back), :]
        dataX.append(a)
    return np.array(dataX)

if __name__=='__main__':
    
    N = 10000
    sigmaf = [0.2, 0.1, 0.05] 
    # sigmaf = [0.59239224, 0.06442296, 0.02609584]
    f0 = [0.0,0.0,0.0]
    f_param = [-0.000279  ,  0.00368725, -0.00037637] 
    # f_param = [0.0214, 0.0231, -0.0225]
    sigma = 0.01498
    plot_inputs = False
    HalfLife = [2.82,  233.38, 1432.94]
    seed_ret = 345
    offset=2
    uncorrelated=True
    hetersk = True
    alpha_h = 0.2
    beta_h = 0.8
    
    rng = np.random.RandomState(seed_ret)
    
    ret, factors, f_speed = ReturnSampler(N, sigmaf, f0, f_param, sigma, plot_inputs, 
                                        HalfLife, rng, offset=offset, adftest = False,
                                        uncorrelated=uncorrelated,
                                        hetersk=hetersk, alpha_h=alpha_h, beta_h=beta_h)
    

    
    
    
    