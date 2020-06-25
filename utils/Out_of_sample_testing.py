# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:21 2020

@author: aless
"""
from tqdm import tqdm
from utils.SimulateData import ReturnSampler
from utils.TradersMarketEnv import MarketEnv
from utils.TradersMarketEnv import ReturnSpace,HoldingSpace
import numpy as np

def Out_sample_test(N_test, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, 
                    seed,Startholding,CostMultiplier,kappa,discount_rate,executeDRL, 
                    executeRL,executeMV,RT,KLM,executeGP,TrainNet,QTable,savedpath,iteration, tag='DQN'):
    
    test_returns, test_factors, test_f_speed = ReturnSampler(N_test, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, seed)
    test_env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                         N_test, discount_rate, f_param, test_f_speed, test_returns, test_factors)
    
    if executeDRL:
        CurrState = test_env.reset()
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