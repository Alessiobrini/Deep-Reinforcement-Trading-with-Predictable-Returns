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


class ReturnSpace(Space):
    def __init__(self, RT: list):
        self.values = np.arange(-RT[0], RT[0] + 1)* RT[1]
        super().__init__(self.values.shape, self.values.dtype)

    def contains(self, x: int):
        return x in self.values

class HoldingSpace(Space):
    def __init__(self, KLM: list):
        self.values = np.arange(-KLM[2], KLM[2] + 1, KLM[1])
        super().__init__(self.values.shape, self.values.dtype)

    def contains(self, x: int):
        return x in self.values    


class ActionSpace(Space):
    def __init__(self, KLM: list):
        self.values = np.arange(-KLM[0], KLM[0] + 1, KLM[1])
        super().__init__(self.values.shape, self.values.dtype)

    def sample(self):
        return np.random.choice(self.values)

    def contains(self, x: int):
        return x in self.values

class CreateQTable():
    def __init__(self, ReturnSpace, HoldingSpace, ActionSpace, tablr, gamma):
        # generate row index of the dataframe with every possible combination
        # of state space variables
        self.ReturnSpace = ReturnSpace
        self.HoldingSpace = HoldingSpace
        self.ActionSpace = ActionSpace
        
        self.tablr = tablr
        self.gamma = gamma
        
        iterables = [self.ReturnSpace.values, self.HoldingSpace.values]
        State_space = pd.MultiIndex.from_product(iterables)
        
        # Create dataframe and set properly index and column heading
        Q_space = pd.DataFrame(index = State_space, columns = self.ActionSpace.values).fillna(0)
        Q_space.index.set_names(['Return','Holding'],inplace=True)
        Q_space.columns.set_names(['Action'],inplace=True)
        # initialize the Qvalues for action 0 as slightly greater than 0 so that
        # 'doing nothing' becomes the default action, instead the default action to be the first column of
        # the dataframe.
        Q_space[0] = 0.0000000001

        self.Q_space = Q_space
    
    
    def getQvalue(self,state):
        ret = state[0]
        holding = state[1]
        return self.Q_space.loc[(ret, holding),]

    def argmaxQ(self,state):
        return self.getQvalue(state).idxmax()

    def getMaxQ(self,state):
        return self.getQvalue(state).max()


    def chooseAction(self,state, epsilon):
        random_action = np.random.rand()
        if (random_action < epsilon):
            # pick one action at random for exploration purposes
            dn = self.ActionSpace.sample()
        else:
            # pick the greedy action
            dn = self.argmaxQ(state)

        return dn    
    
    def chooseGreedyAction(self,state):
        return self.argmaxQ(state) 

    def update(self,DiscrCurrState,DiscrNextState,shares_traded,Result):
        q_sa = self.Q_space.loc[tuple(DiscrCurrState), shares_traded]
        increment = self.tablr * ( Result['Reward_Q'] + \
                              self.gamma * self.getMaxQ(DiscrNextState) - q_sa)
        self.Q_space.loc[tuple(DiscrCurrState), shares_traded] = q_sa + increment

    
    def save(self, savedpath, N_train):
        self.Q_space.columns = [str(c) for c in self.Q_space.columns]
        self.Q_space.to_parquet(os.path.join(savedpath,
                              'QTable' + format_tousands(N_train) 
                              + '.parquet.gzip'),compression='gzip')
    

class MarketEnv(gym.Env):
    '''Custom  Market Environment that follows gym interface'''
    

    def __init__(self,
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
                 factors: Union[list or np.ndarray],):
        
        super(MarketEnv, self).__init__()
        
        self.HalfLife = HalfLife
        self.Startholding = Startholding
        self.sigma = sigma
        self.CostMultiplier = CostMultiplier
        self.kappa = kappa
        self.N_train = N_train
        self.discount_rate = discount_rate
        self.f_param = f_param
        self.f_speed = f_speed
        self.returns = returns
        self.factors = factors
        #only for TabQ
        # self.ReturnSpace = ReturnSpace
        # self.HoldingSpace = HoldingSpace

        
        colnames = ['returns'] + ['factor_' + str(hl) for hl in HalfLife]
        # self.colnames = colnames
        res_df = pd.DataFrame(np.concatenate([np.array(self.returns).reshape(-1,1),
                                              np.array(self.factors)],axis=1),columns = colnames)
        
        res_df = res_df.astype(np.float32)
        #res_df['Copy_Target'] = np.nan
        self.res_df = res_df

    
    def step(self, currState: Union[Tuple or np.ndarray],shares_traded: int, iteration: int):
        nextFactors = self.factors[iteration + 1]
        nextRet = self.returns[iteration + 1]
        nextHolding = currState[1] + shares_traded
        nextState = np.array([nextRet, nextHolding])
        
        Result = self.getreward(currState, nextState, 'DQN')
        
        return nextState, Result, nextFactors
       
    def reset(self):
        currState = np.array([self.returns[0],self.Startholding])
        return currState
    
    
    def discrete_step(self, discretecurrState: Union[Tuple or np.ndarray],shares_traded: int, iteration: int):
        discretenextRet = self.find_nearest_return(self.returns[iteration + 1])
        discretenextHolding = self.find_nearest_holding(discretecurrState[1] + shares_traded)
        discretenextState = np.array([discretenextRet, discretenextHolding])
        Result = self.getreward(discretecurrState, discretenextState, 'Q')
        return discretenextState, Result
       
    def discrete_reset(self):
        discretecurrState = np.array([self.find_nearest_return(self.returns[0]),
                                      self.find_nearest_holding(self.Startholding)])
        return discretecurrState
    
    def find_nearest_return(self, value):
        array = np.asarray(self.returns_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def find_nearest_holding(self, value):
        array = np.asarray(self.holding_space.values)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
             
    def totalcost(self,shares_traded: Union[float or int]) -> Union[float or int]:

        Lambda = self.CostMultiplier * self.sigma**2
        quadratic_costs = 0.5 * (shares_traded**2) * Lambda
        
        return quadratic_costs

    
    # REWARD FUNCTIONS
    
    def getreward(self, 
                  currState: Tuple[Union[float or int],Union[float or int]],
                  nextState: Tuple[Union[float or int],Union[float or int]],
                  tag: str) -> dict:

        
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
                  'CurrHolding_{}'.format(tag): currHolding,
                  'NextHolding_{}'.format(tag): nextHolding,
                  'Action_{}'.format(tag): shares_traded,
                  'GrossPNL_{}'.format(tag): GrossPNL,
                  'NetPNL_{}'.format(tag): NetPNL,
                  'Risk_{}'.format(tag): Risk,
                  'Cost_{}'.format(tag): Cost,
                  'Reward_{}'.format(tag) : Reward,
                  }
        return Result
    

    def store_results(self,
                      Result:dict,
                      iteration: int):

        if iteration==0:
            for key in Result.keys(): 
                self.res_df[key] = 0.0
                self.res_df.at[iteration,key] = Result[key]
            self.res_df = self.res_df.astype(np.float32)
        else:
            for key in Result.keys(): 
                self.res_df.at[iteration,key] = Result[key]



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
    
    
    def save_outputs(self, savedpath, test= None):
        
        if not test:
            self.res_df.to_parquet(os.path.join(savedpath,
                                  'Results_' + format_tousands(self.N_train) 
                                  + '.parquet.gzip'),compression='gzip')
        else:
            self.res_df.to_parquet(os.path.join(savedpath,
                                  'TestResults_' + format_tousands(self.N_train) 
                                  + '.parquet.gzip'),compression='gzip')
        
