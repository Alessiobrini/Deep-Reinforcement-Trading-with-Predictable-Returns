# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:21 2020

@author: aless
"""
from tqdm import tqdm
from utils.SimulateData import ReturnSampler
from utils.MarketEnv import MarketEnv, RecurrentMarketEnv
from utils.MarketEnv import ReturnSpace,HoldingSpace
from utils.SimulateData import create_lstm_tensor
import numpy as np
import pandas as pd
from typing import Union, Optional
from pathlib import Path

def Out_sample_test(N_test : int,
                    sigmaf : Union[float or list or np.ndarray],
                    f0 : Union[float or list or np.ndarray],
                    f_param: Union[float or list or np.ndarray],
                    sigma: Union[float or list or np.ndarray],
                    plot_inputs: int,
                    HalfLife: Union[int or list or np.ndarray],
                    Startholding: Union[float or int],
                    CostMultiplier: float,
                    kappa: float,
                    discount_rate: float,
                    executeDRL: bool, 
                    executeRL: bool,
                    executeMV: bool,
                    RT: list,
                    KLM: list,
                    executeGP: bool,
                    TrainNet,
                    savedpath: Union[ str or Path],
                    iteration: int,
                    recurrent_env: bool = False,
                    unfolding: int = 1,
                    QTable: Optional[pd.DataFrame] = None,
                    seed: int = None,
                    action_limit=None, 
                    tag='DQN'):
    
    test_returns, test_factors, test_f_speed = ReturnSampler(N_test, sigmaf, f0, f_param, sigma, 
                                                             plot_inputs, HalfLife, seed,
                                                             offset=unfolding + 1)
    if recurrent_env:
        test_returns_tens = create_lstm_tensor(test_returns.reshape(-1,1), unfolding)
        test_factors_tens = create_lstm_tensor(test_factors, unfolding)
        test_env = RecurrentMarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa,  N_test, discount_rate, 
                             f_param, test_f_speed, test_returns, test_factors, test_returns_tens, test_factors_tens, action_limit)  
    else:
        test_env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa,  N_test, discount_rate, 
                             f_param, test_f_speed, test_returns, test_factors, action_limit)
    
    if executeDRL:
        CurrState, _ = test_env.reset()
    if executeRL:
        test_env.returns_space = ReturnSpace(RT)
        test_env.holding_space = HoldingSpace(KLM)
        DiscrCurrState = test_env.discrete_reset()
    if executeGP:
        CurrOptState = test_env.opt_reset()
        OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()  
    if executeMV:
        CurrMVState = test_env.opt_reset()
    
    for i in tqdm(iterable=range(N_test + 1), desc='Testing DQNetwork'):
        if executeDRL:
            if tag == 'DQN':
                shares_traded = TrainNet.greedy_action(CurrState)
                NextState, Result, NextFactors = test_env.step(CurrState, shares_traded, i)
                test_env.store_results(Result, i)
            elif tag == 'DDPG':
                shares_traded = TrainNet.p_model(np.atleast_2d(CurrState.astype('float32')),training=False)
                NextState, Result, NextFactors = test_env.step(CurrState, shares_traded, i, tag=tag)
                test_env.store_results(Result, i)
            CurrState = NextState

        if executeRL:
            shares_traded = int(QTable.chooseGreedyAction(DiscrCurrState))
            DiscrNextState, Result = test_env.discrete_step(DiscrCurrState, shares_traded, i)
            test_env.store_results(Result, i)
            DiscrCurrState = DiscrNextState
        
        if executeGP:
            NextOptState, OptResult = test_env.opt_step(CurrOptState, OptRate, DiscFactorLoads, i)
            test_env.store_results(OptResult, i) 
            CurrOptState = NextOptState
            
        if executeMV:
            NextMVState, MVResult = test_env.mv_step(CurrMVState, i)
            test_env.store_results(MVResult, i) 
            CurrMVState = NextMVState

    test_env.save_outputs(savedpath, test=True, iteration=iteration)