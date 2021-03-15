# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:38:05 2020

@author: aless
"""
import pdb
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
import pandas as pd
from statsmodels.regression.linear_model import OLS
from utils.env import MarketEnv
from utils.math_tools import boltzmann, unscale_action
from scipy.stats import norm


def get_action_boundaries(
    HalfLife: Union[int or list or np.ndarray],
    Startholding: Union[int or float],
    sigma: float,
    CostMultiplier: float,
    kappa: float,
    N_train: int,
    discount_rate: float,
    f_param: Union[float or list or np.ndarray],
    f_speed: Union[float or list or np.ndarray],
    returns: Union[list or np.ndarray],
    factors: Union[list or np.ndarray],
    qts: list = [0.01, 0.99],
    min_n_actions: bool = True,
    experiment_type: str = 'synth',
):

    """
    Function to compute heuristically the boundary of discretized spaces
    of Q-learning and DQN in order to be comparable with the benchmark solution

    Parameters
    ----------
    HalfLife: Union[int or list or np.ndarray]
        List of HalfLife of mean reversion when the simulated dynamic is driven by
        factors

    Startholding: Union[int or float]
        Initial portfolio holding, usually set at 0

    sigma: float
        Constant volatility for the simulated returns of the asset

    CostMultiplier: float
        Transaction cost parameter which regulates the market liquidity

    kappa: float
        Risk averion parameter

    N_train: int
        Length of simulated experiment

    discount_rate: float
        Discount rate for the reward function

    f_param: Union[float or list or np.ndarray]
        List of factor loadings when the simulated dynamic is driven by
        factors

    f_speed: Union[float or list or np.ndarray]
        List of speed of mean reversion when the simulated dynamic is driven by
        factors

    returns: Union[list or np.ndarray]
        Array of simulated returns

    factors: Union[list or np.ndarray]
        Array of simulated factors when the simulated dynamic is driven by
        factors or lagged factors when it is not the case

    qts: list
        Quantiles to bound the discretized state space acccording to the
        performed action distribution of the benchmark solution

    min_n_actions: bool
        Boolean to regulate if the number of discretized action is minimum
        (just a long,buy and hold action) of if there are more actions available
        TODO implement more than 5 actions possible here

    Returns
    ----------
    dfRetLag: pd.DataFrame
        Output dataframe which contains the original series and the lagged series

    """

    env = MarketEnv(
        HalfLife,
        Startholding,
        sigma,
        CostMultiplier,
        kappa,
        N_train,
        discount_rate,
        f_param,
        f_speed,
        returns,
        factors,
    )

    CurrOptState = env.opt_reset()
    OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()

    cycle_len = len(returns) - 1
    for i in tqdm(iterable=range(cycle_len), desc="Selecting Action boundaries"):
        NextOptState, OptResult = env.opt_step(
            CurrOptState, OptRate, DiscFactorLoads, i
        )
        env.store_results(OptResult, i)
        CurrOptState = NextOptState

    action_quantiles = env.res_df["OptNextAction"].quantile(qts).values
    
    if experiment_type == 'real':
        qt = np.max(np.abs(action_quantiles))
        length = len(str(int(np.round(qt))))
        action = int(np.abs(np.round(qt, -length + 2)))
    else:
        qt = np.min(np.abs(action_quantiles))
        length = len(str(int(np.round(qt))))
        action = int(np.abs(np.round(qt, -length + 1)))

    if min_n_actions:
        action_ranges = [action, action]
    else:
        action_ranges = [
            action,
            int(action / 2),
        ]  # TODO modify for allowing variables range of actions

    ret_range = float(max(np.abs(returns.min()), returns.max()))

    holding_quantiles = env.res_df["OptNextHolding"].quantile(qts).values
    if np.abs(holding_quantiles[0]) - np.abs(holding_quantiles[1]) < 1000:
        holding_ranges = int(np.abs(np.round(holding_quantiles[0], -2)))
    else:
        holding_ranges = int(np.round(np.min(np.abs(holding_quantiles)), -2))

    return action_ranges, ret_range, holding_ranges


def CalculateLaggedSharpeRatio(
    series: Union[pd.Series or np.ndarray],
    lags: list,
    nameTag: str,
    seriestype: str = "price",
) -> pd.DataFrame:
    """
    Function which accepts a return or a price series and compute lagged variables
    according to the list 'lags'

    Parameters
    ----------
    series: Union[pd.Series or np.ndarray]
        Return or price series to compute lagged variables

    lags: list
        List of lags to compute

    nameTag: str
        Name of the series for the DataFrame columns

    seriestype: str
        Type of input series. It can be 'price' or 'return'

    Returns
    ----------
    dfRetLag: pd.DataFrame
        Output dataframe which contains the original series and the lagged series

    """
    # preallocate Df to store the shifted returns.
    dfRetLag = pd.DataFrame()
    if seriestype == "price":
        dfRetLag[nameTag] = series.pct_change(periods=1)
    elif seriestype == "return":
        dfRetLag[nameTag] = series
    # loop for lags calculate returns and shifts the series.
    for value in lags:
        # Calculate returns using the lag and shift dates to be used in the regressions.
        if value != 1:
            dfRetLag[nameTag + "_" + str(value)] = (
                dfRetLag[nameTag].shift(1).rolling(window=value).mean()
                / dfRetLag[nameTag].shift(1).rolling(window=value).std()
            )
        else:
            dfRetLag[nameTag + "_" + str(value)] = dfRetLag[nameTag].shift(1)

    dfRetLag.dropna(inplace=True)

    return dfRetLag


def RunModels(
    y: Union[pd.Series or pd.DataFrame],
    X: Union[pd.Series or pd.DataFrame],
    mr_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:  # TODO add instantiation of statsmodel output
    """
    Funtion which estimate speeds of mean reversion and factor loadings by
    using OLS

    Parameters
    ----------
    y: Union[pd.Series or pd.DataFrame]
        Dependent variable of the regressions

    y: Union[pd.Series or pd.DataFrame]
        Explanatory variables of the regressions

    mr_only: bool
        Boolean to regulate if fitting of factor loadings is required or not.
        If True, it fits only the mean reversion parameters


    Returns
    ----------
    params_retmodel: np.ndarray
        Array of fitted factor loading

    params_meanrev: np.ndarray
        Array of fitted speed of mean reversion

    fitted_retmodel
        Stasmodel output of the fit for the factor loadings

    fitted_ous
        Stasmodel output of the fit for the speeds fo mean reversion

    """

    if mr_only:
        params_meanrev = {}
        fitted_ous = []
        # model for mean reverting equations
        for col in X.columns:
            ou = OLS(X[col].diff(1).dropna(), X[col].shift(1).dropna())
            fitted_ou = ou.fit(cov_type="HC0")
            fitted_ous.append(fitted_ou)
            params_meanrev["params_{}".format(col)] = np.array(fitted_ou.params)
            # params_meanrev['pval' + col] = fitted_ou.pvalues

        return params_meanrev, fitted_ous

    params_retmodel = {}
    # model for returns
    retmodel = OLS(y, X)
    fitted_retmodel = retmodel.fit(cov_type="HC0")
    # store results
    params_retmodel["params"] = np.array(fitted_retmodel.params)
    params_retmodel["pval"] = np.array(fitted_retmodel.pvalues)

    params_meanrev = {}
    fitted_ous = []
    # model for mean reverting equations
    for col in X.columns:
        ou = OLS(X[col].diff(1).dropna(), X[col].shift(1).dropna())
        fitted_ou = ou.fit(cov_type="HC0")
        fitted_ous.append(fitted_ou)
        params_meanrev["params" + col] = np.array(fitted_ou.params)
        # params_meanrev['pval' + col] = fitted_ou.pvalues

    return params_retmodel, params_meanrev, fitted_retmodel, fitted_ous



def get_bet_size(qvalues: np.ndarray,side_action: float,action_limit: float, 
                 zero_action: bool, rng, discretization: float = None,
                 temp: float = 200.0) -> float:
    """
    Get the size of the bet by using qvalues of DQN as probabilities. In principle
    a continuous range of action inside a boundary is outputted, but there is also
    an option to discretize the range.

    Parameters
    ----------
    qvalues: Union[pd.Series or pd.DataFrame]
        Array of qvalues

    side_action: Union[pd.Series or pd.DataFrame]
        Action taken by the algorithm which specifies the size

    action_limit: bool
        Lower and upper boundary for the action space
    
    zero_action: bool
        Regulate the presence of hold actions (no trade)
    
    rng: 
        Random number generator
    
    discretization: float
        Level of discretization. If none, no discretization will be applied
        
    temp: float
        Temperature of boltzmann equation

    Returns
    ----------
    size_action: float
        Action which reflect the size of the bet in addition the provided side

    """ 
    # when qvalues are not provided because the action is taked at random
    if qvalues == None:
        if side_action == 0.0:
            m = 0
        elif side_action == -1.0:
            m = rng.uniform(-1.0,0.0)
        elif side_action == 1.0:
            m = rng.uniform(0.0,1.0)
        if discretization:
            m = np.round(m/discretization,0)*discretization
        size_action = unscale_action(action_limit, m)
    
    else:

        # one hot vector of qvalues by the max action
        if zero_action:
            n_actions = 3
        else:
            n_actions = 2
            
        idx = tf.one_hot(tf.math.argmax(qvalues,axis=1), n_actions)
        
        # get prob by using boltzmann
        prob = boltzmann(qvalues,T=temp)
        act_prob = tf.math.reduce_sum(prob * idx, axis=1)
        # avoid division by 0 and get z statistics
        z = (act_prob - (1/n_actions)) / (tf.math.sqrt(act_prob * (1-act_prob)) + 0.00000007)
        # get size in the range -1,1
        m = (2*norm.cdf(z) - 1) * side_action
        if discretization:
            m = np.round(m/discretization,0)*discretization
        # rescale action to the proper range
        size_action = float(unscale_action(action_limit, m))
    
    return size_action