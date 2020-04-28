# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:39:12 2020

@author: aless
"""
# following https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

from typing import Union, Tuple
import pandas as pd
import numpy as np
import gym
from gym import spaces
from gym.spaces.space import Space
import pdb, os
from utils.format_tousands import format_tousands

class ActionSpace(Space):
    def __init__(self, KL: list):
        self.values = np.arange(-KL[0], KL[0] + 1, KL[1])
        super().__init__(self.values.shape, self.values.dtype)

    def sample(self):
        return np.random.choice(self.values)

    def contains(self, x: int):
        return x in self.values


class MarketEnv(gym.Env):
    '''Custom  Market Environment that follows gym interface'''
    

    def __init__(self,
                 HalfLife: Union[int or list or np.ndarray],
                 Startholding: Union[int or float],
                 sigma: float,
                 CostMultiplier: float,
                 kappa: float,
                 N_train: int,
                 plot_insample: Union[int or bool],
                 discount_rate: float,
                 f_param: Union[float or list or np.ndarray],
                 f_speed: Union[float or list or np.ndarray],
                 returns: Union[list or np.ndarray], 
                 factors: Union[list or np.ndarray]):
        
        super(MarketEnv, self).__init__()
        
        self.HalfLife = HalfLife
        self.Startholding = Startholding
        self.sigma = sigma
        self.CostMultiplier = CostMultiplier
        self.kappa = kappa
        self.N_train = N_train
        self.plot_insample = plot_insample
        self.discount_rate = discount_rate
        self.f_param = f_param
        self.f_speed = f_speed
        self.returns = returns
        self.factors = factors
        
        colnames = ['returns'] + ['factor_' + str(hl) for hl in HalfLife]
        # self.colnames = colnames
        res_df = pd.DataFrame(np.concatenate([np.array(self.returns).reshape(-1,1),
                                              np.array(self.factors)],axis=1),columns = colnames)
        
        res_df = res_df.astype(np.float32)
        self.res_df = res_df
        #self.res_dict = {}
    
    def step(self, currState: Union[Tuple or np.ndarray],shares_traded: int, iteration: int):
        nextRet = self.returns[iteration + 1]
        nextHolding = currState[1] + shares_traded
        nextState = np.array([nextRet, nextHolding])
        
        Result = self.getreward(currState, nextState)
        
        return nextState, Result
       
    def reset(self):
        currState = np.array([self.returns[0],self.Startholding])
        return currState
             
    def totalcost(self,shares_traded: Union[float or int]) -> Union[float or int]:

        Lambda = self.CostMultiplier * self.sigma**2
        quadratic_costs = 0.5 * (shares_traded**2) * Lambda
        
        return quadratic_costs

    
    # REWARD FUNCTIONS
    
    def getreward(self, 
                  currState: Tuple[Union[float or int],Union[float or int]],
                  nextState: Tuple[Union[float or int],Union[float or int]]) -> dict:

        
        # Remember that a state is a tuple (price, holding)
        currRet = currState[0]
        nextRet = nextState[0]
        currHolding = currState[1]
        nextHolding = nextState[1]
        
        shares_traded = nextHolding - currHolding
        GrossPNL = nextHolding * nextRet
        Risk = 0.5 * self.kappa * ((nextHolding**2) * (self.sigma**2))
        Cost = self.totalcost(shares_traded)
        NetPNL = GrossPNL - Cost   
        Reward = GrossPNL - Risk - Cost
        
        Result = {
                  'CurrHolding': currHolding,
                  'NextHolding': nextHolding,
                  'Action': shares_traded,
                  'GrossPNL': GrossPNL,
                  'NetPNL': NetPNL,
                  'Risk': Risk,
                  'Cost': Cost,
                  'Reward' : Reward,
                  }
        return Result
    

    def store_results(self,
                      Result:dict,
                      iteration: int):
        #pdb.set_trace()
        if iteration==0:
            for key in Result.keys(): 
                self.res_df[key] = 0.0
                self.res_df.at[iteration,key] = Result[key]
                # if CurrQ:
                #     self.res_df['Q'] = 0.0
                #     self.res_df.at[iteration,'Q'] = CurrQ
            #pdb.set_trace()
            self.res_df = self.res_df.astype(np.float32)
        else:
            for key in Result.keys(): 
                #pdb.set_trace()
                self.res_df.at[iteration,key] = Result[key]
                # if CurrQ:
                #     self.res_df.at[iteration,'Q'] = CurrQ
        # if iteration==0:
        #     self.colnames.extend(list(Result.keys()))
                       
        #     if iteration in self.res_dict.keys():
        #         self.res_dict[iteration].extend(list(Result.values()))
        #     else:
        #         self.res_dict[iteration] = list(Result.values())
        # else:
            
        #     if iteration in self.res_dict.keys():
        #         self.res_dict[iteration].extend(list(Result.values()))
        #     else:
        #         self.res_dict[iteration] = list(Result.values())
    
    def opt_reset(self):
        
        currOptState = np.array([self.returns[0],self.factors[0],self.Startholding])
        return currOptState
        
            
    def opt_trading_rate_disc_loads(self):
        
        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(- self.discount_rate/260)  
        
        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = (self.kappa * ( 1 - rho) + self.CostMultiplier *rho)
        num2 = np.sqrt(num1**2 + 4 * self.kappa * self.CostMultiplier * (1 - rho)**2)
        den = 2* (1 - rho)
        a = (-num1 + num2)/ den
        
        OptRate = a / self.CostMultiplier
        
        DiscFactorLoads = self.f_param / (1 + self.f_speed * ((OptRate * self.CostMultiplier) / \
                                                                 self.kappa))
    
        return OptRate, DiscFactorLoads
    
    def opt_step(self, 
                 currOptState: Tuple, 
                 OptRate: float,
                 DiscFactorLoads: np.ndarray,
                 iteration: int) -> dict:
        
        
        #CurrReturns = currOptState[0]
        CurrFactors = currOptState[1]
        OptCurrHolding = currOptState[2]
           
        # Optimal traded quantity between period
        OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * \
                      (1/(self.kappa * (self.sigma)**2)) * \
                       np.sum(DiscFactorLoads * CurrFactors)
                       
        # Traded quantity as for the Markovitz framework  (Mean-Variance framework)            
        MVNextHolding =  (1/(self.kappa * (self.sigma)**2)) * \
                        np.sum(self.f_param * CurrFactors)
                       
        nextReturns = self.returns[iteration + 1]
        nextFactors = self.factors[iteration + 1]
        nextOptState = (nextReturns, nextFactors, OptNextHolding)
        
        OptResult = self.get_opt_reward(currOptState, nextOptState, MVNextHolding)
        
        return nextOptState,OptResult
        
    def get_opt_reward(self,
                       currOptState: Tuple[Union[float or int],Union[float or int]],
                       nextOptState: Tuple[Union[float or int],Union[float or int]],
                       MVNextHolding) -> dict:
        
        # Remember that a state is a tuple (price, holding)
        #currRet = currOptState[0]
        nextRet = nextOptState[0]
        OptCurrHolding = currOptState[2]
        OptNextHolding = nextOptState[2]
        
        
        # Traded quantity between period
        OptNextAction = OptNextHolding - OptCurrHolding
        # Portfolio variation
        OptGrossPNL = OptNextHolding * nextRet #currRet
        # Risk
        OptRisk = 0.5 * self.kappa * ((OptNextHolding)**2 * (self.sigma)**2)
        # Transaction costs
        OptCost = self.totalcost(OptNextAction)
        # Portfolio Variation including costs
        OptNetPNL = OptGrossPNL - OptCost
        # Compute reward    
        OptReward = OptGrossPNL - OptRisk - OptCost
        
        # Store quantities
        Result = {
                  'OptNextAction': OptNextAction,
                  'OptNextHolding': OptNextHolding,
                  'OptGrossPNL': OptGrossPNL,
                  'OptNetPNL': OptNetPNL,
                  'OptRisk': OptRisk,
                  'OptCost': OptCost,
                  'OptReward' : OptReward,
                  'MVNextHolding' : MVNextHolding
                  }
        
        return Result
    
    
    def save_outputs(self, savedpath):
        
        # res_df = pd.DataFrame.from_dict(self.res_dict,orient='index', columns = self.colnames[2:])
        # res_df['Returns'] = self.returns[:-1]
        # res_df['Factors'] = self.factors[:-1]
        # self.res_df = res_df
        #pdb.set_trace()
        self.res_df.to_parquet(os.path.join(savedpath,
                              'Results_' + format_tousands(self.N_train) 
                              + '.parquet.gzip'),compression='gzip')
        
        # pdb.set_trace()
        # self.res_df.to_parquet(os.path.join(savedpath,
        #                       'Results_' + format_tousands(self.N_train) 
        #                       + '.parquet.gzip'),compression='gzip')
