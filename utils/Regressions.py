# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:50:09 2020

@author: aless
"""
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS,GLS
import statsmodels.api as sm
import pdb

def CalculateLaggedSharpeRatio(series, lags, nameTag):    
    # preallocate Df to store the shifted returns.
    dfRetLag = pd.DataFrame()
    dfRetLag[nameTag] = series.pct_change(periods = 1)
    # loop for lags calculate returns and shifts the series. 
    for value in lags:
        # Calculate returns using the lag and shift dates to be used in the regressions. 
        dfRetLag[nameTag + '_' + str(value)] = (dfRetLag[nameTag].shift(1).rolling(window = value).mean() / 
                                                dfRetLag[nameTag].shift(1).rolling(window = value).std())
    
    dfRetLag.dropna(inplace=True)
    
    return dfRetLag

def RunModels(y,X):
    
    params_retmodel = {}
    # model for returns
    retmodel = OLS(y,X)
    fitted_retmodel = retmodel.fit(cov_type='HC0')
    # store results
    params_retmodel['params'] = np.array(fitted_retmodel.params)
    params_retmodel['pval'] = np.array(fitted_retmodel.pvalues)
    
    params_meanrev = {}
    fitted_ous = []
    # model for mean reverting equations
    for col in X.columns:
        ou = OLS(X[col].diff(1).dropna(),X[col].shift(1).dropna())
        fitted_ou = ou.fit(cov_type='HC0')
        fitted_ous.append(fitted_ou)
        params_meanrev['params' + col] = np.array(fitted_ou.params)
        #params_meanrev['pval' + col] = fitted_ou.pvalues
        
    return params_retmodel, params_meanrev, fitted_retmodel, fitted_ous