# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:00:44 2019

@author: aless
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import seaborn 
seaborn.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['savefig.dpi'] = 90
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

from statsmodels.tsa.stattools import adfuller

import sys
import pdb
import pickle
import os
import shutil
from tqdm import tqdm

from utils.format_tousands import format_tousands
from utils.save_config import SaveConfig


class QTraderObject(object):
    
    '''
    This class is useful to implement a Q-learning agent that produces and compare the simple 
    synthetic experiments as in Ritter (2017) and the Dynamic Programming analytical solution of
    Garleanu and Pedersen (2013).
    '''

    # ----------------------------------------------------------------------------
    # Init method      
    # ----------------------------------------------------------------------------
    def __init__(self, Param):

        '''
        init method to initialize the class. Parameter inputs are stored 
        as properties of the object.
        '''
        self.Param = Param
        self._setSpaces(Param)
        
    # ----------------------------------------------------------------------------
    # Private method      
    # ----------------------------------------------------------------------------
    def _setSpaces(self, Param):
        '''
        Create discrete action, holding and price spaces
        '''
        ParamSpace = { 
                      'A_space' : np.arange(-Param['K'] ,Param['K']+1, Param['LotSize']),
                      'H_space' : np.arange(-Param['M'],Param['M']+1, Param['LotSize']),
                      'R_space' : np.arange(-Param['R_min'],Param['R_max']+1)*Param['TickSize']
                      }

        self.ParamSpace = ParamSpace
    # ----------------------------------------------------------------------------
    # Public method       
    # ----------------------------------------------------------------------------
    
    # ROUNDING FUNCTIONS
    
    def find_nearest_return(self, value):
        '''
        Function to ensure that prices don't exit the valid range
        '''
        array = np.asarray(self.ParamSpace['R_space'])
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    def find_nearest_holding(self, value):
        '''
        Function to ensure that holdings don't exit the valid range
        '''
        array = np.asarray(self.ParamSpace['H_space'])
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    # PRICE SAMPLERS
    
    def ReturnSampler(self, sampleSize, seed=None):
      
        '''
        Simulate a price process according to a selected dynamic (Ornstein-Uhlenbeck)
        '''
        # Set seed to make the out-of-sample experiment reproducible
        np.random.seed(seed)
        
        eps = np.random.randn(sampleSize + 2) 
        sigmaf = self.Param['sigmaf']
        lambdas = np.around(np.log(2)/self.Param['HalfLife'],4)
        
        # if condition to enable different simulations
        if self.Param['NumFactors'] == 1:
            # simulate the single factor according to OU process
            lambda_5D = lambdas[0]
            f0_5D = self.Param['f_init'][0]
            f5D = []
       
            for i in tqdm(iterable=range(sampleSize + 2), desc='Simulating Factors'):
                # first factor
                f1_5D = (1 - lambda_5D) * f0_5D + sigmaf * eps[i]
                f5D.append(f1_5D)
                f0_5D = f1_5D

            factors  = np.array(f5D) 
            #pdb.set_trace()
            # instantiate factor parameter for regression
            f_param = np.array(self.Param['f_param'])[0]

            # calculate return series depeding on boolean for adding noise
            if self.Param['add_noise']:
                
                u = np.random.randn(sampleSize + 2)
                sigma = self.Param['sigma']
                realret = [(f_param * factors[i]) 
                           + sigma * u[i] for i in range(sampleSize + 2)]
            else:
                realret = [(f_param * factors[i]) for i in range(sampleSize + 2)]

            capret = [self.find_nearest_return(r) for r in realret]
                        
            self.f_speed = np.array([lambda_5D])

            

        elif self.Param['NumFactors'] == 2:
            
            # simulate factors according to OU process
            lambda_5D = lambdas[0]
            lambda_1Y = lambdas[1]
     
            f0_5D = self.Param['f_init'][0]
            f5D = []
            f0_1Y = self.Param['f_init'][1] 
            f1Y = []

            
            # suppose that they are correctly simulated (TODO : check starting value)
            for i in tqdm(iterable = range(sampleSize + 2), desc='Simulating Factors'):
                # first factor
                f1_5D = (1 - lambda_5D) * f0_5D + sigmaf * eps[i]
                f5D.append(f1_5D)
                f0_5D = f1_5D
                # second factor
                f1_1Y = (1 - lambda_1Y) * f0_1Y + sigmaf * eps[i]
                f1Y.append(f1_1Y)
                f0_1Y = f1_1Y        
            
            # concatenate factors into an unique array
            factors = np.concatenate([np.array(f5D).reshape(-1,1),
                                      np.array(f1Y).reshape(-1,1)],axis=1)
            
            # instantiate factor parameters for regression
            f_param = np.array(self.Param['f_param'])[:2]
            
            
            # calculate return series depeding on boolean for adding noise
            if self.Param['add_noise']:
                
                u = np.random.randn(sampleSize + 2)
                sigma = self.Param['sigma']
                realret = [(f_param @ factors[i]) 
                           + sigma * u[i] for i in range(sampleSize + 2)]
            else:
                realret = [(f_param @ factors[i]) for i in range(sampleSize + 2)]            
            
            capret = [self.find_nearest_return(r) for r in realret]
            
            
            self.f_speed = np.array([lambda_5D,lambda_1Y])
            
        elif self.Param['NumFactors'] == 3:

         
            # simulate factors according to OU process
            lambda_5D = lambdas[0]
            lambda_1Y = lambdas[1]
            lambda_5Y = lambdas[2]
            
            f0_5D = self.Param['f_init'][0]
            f5D = []
            f0_1Y = self.Param['f_init'][1] 
            f1Y = []
            f0_5Y = self.Param['f_init'][2] 
            f5Y = []
            
            # suppose that they are correctly simulated (TODO : check starting value)
            for i in tqdm(iterable = range(sampleSize + 2), desc='Simulating Factors'):
                # first factor
                f1_5D = (1 - lambda_5D) * f0_5D + sigmaf * eps[i]
                f5D.append(f1_5D)
                f0_5D = f1_5D
                # second factor
                f1_1Y = (1 - lambda_1Y) * f0_1Y + sigmaf * eps[i]
                f1Y.append(f1_1Y)
                f0_1Y = f1_1Y
                # third factor
                f1_5Y = (1 - lambda_5Y) * f0_5Y + sigmaf * eps[i]
                f5Y.append(f1_5Y)
                f0_5Y = f1_5Y
            
            
            # concatenate factors into an unique array
            factors = np.concatenate([np.array(f5D).reshape(-1,1),
                                      np.array(f1Y).reshape(-1,1),
                                      np.array(f5Y).reshape(-1,1)],axis=1)
            
            # instantiate factor parameters for regression
            f_param = np.array(self.Param['f_param'])
            
            # calculate return series depeding on boolean for adding noise
            if self.Param['add_noise']:
                
                u = np.random.randn(sampleSize + 2)
                sigma = self.Param['sigma']
                realret = [(f_param @ factors[i]) 
                           + sigma * u[i] for i in range(sampleSize + 2)]
            else:
                realret = [(f_param @ factors[i]) for i in range(sampleSize + 2)]  
            
            capret = [self.find_nearest_return(r) for r in realret]
                
            
            self.f_speed = np.array([lambda_5D,lambda_1Y,lambda_5Y])
    
        
        else:
            
            print('##############################################################################')
            print('NumFactors greater than 3! Check the dimensionality or implement more factors!')
            print('##############################################################################')
            
        print(str(len(self.f_speed)) + ' factor(s) simulated')
        print('max realret ' + str(max(realret)))
        print('min realret ' + str(min(realret)))
        print('################################################################')
        print('max capret ' + str(max(capret)))
        print('min capret ' + str(min(capret)))
        print('################################################################')
        print(self.ParamSpace['R_space'])
        
        # plots for factor, returns and prices
        if self.Param['plot_inputs']:
            fig1 = plt.figure()
            fig2 = plt.figure()

            ax1 = fig1.add_subplot(111)
            ax2 = fig2.add_subplot(111)
                
            ax1.plot(factors)
            ax1.legend(['5D','1Y','5Y'])
            ax1.set_title('Factors')
            ax2.plot(capret)
            ax2.plot(realret,alpha=0.7)
            plt.legend(['CapReturns','Returns'])
            ax2.set_title('Returns')
            

            # test=adfuller(realret)
            # print('Test ADF for generated return series')
            # print("Test Statistic: " + str(test[0]))
            # print("P-value: " + str(test[1]))
            # print("Used lag: " + str(test[2]))
            # print("Number of observations: " + str(test[3]))
            # print("Critical Values: " + str(test[4]))
            # print("AIC: " + str(test[5]))
            
            sys.exit()
        
        
        return realret,capret,factors


    # COST FUNCTIONS
    
    def TotalCost(self,dn):
        '''
        Calculate quadratic transaction costs 
        '''
        
        # calculate Kyle's Lambda (Garleanu Pedersen 2013)
        Lambda = self.Param['CostMultiplier'] * self.Param['sigma']**2
        quadratic_costs = 0.5 * (dn**2) * Lambda
        
        return quadratic_costs

    
    # REWARD FUNCTIONS
    
    def GetReward(self, currState, nextState):

        '''
        Compute the one step reward
        '''
        
        # Remember that a state is a tuple (price, holding)
        currRet = currState[0]
        nextRet = nextState[0]
        currHolding = currState[1]
        nextHolding = nextState[1]
        
        #pdb.set_trace()
        # Traded quantity between period
        dn = nextHolding - currHolding
        # Portfolio variation
        GrossPNL = nextHolding * nextRet
        # Risk
        Risk = 0.5 * self.Param['kappa'] * ((nextHolding**2) * (self.Param['sigma']**2))
        # Transaction costs
        Cost = self.TotalCost(dn)
        # Portfolio Variation including costs
        NetPNL = GrossPNL - Cost
        # Compute reward    
        Reward = GrossPNL - Risk - Cost
        
        #pdb.set_trace()
        
        # Store quantities
        Result = {
                  'CurrHolding': currHolding,
                  'NextHolding': nextHolding,
                  'Action': dn,
                  'GrossPNL': GrossPNL,
                  'NetPNL': NetPNL,
                  'Risk': Risk,
                  'Cost': Cost,
                  'Reward' : Reward,
                  }
        return Result


    # Q TABLE GENERATOR
    
    def CreateQTable(self):
        '''
        Create the QTable for all possible state-action cases 

        '''
        iterables = [self.ParamSpace['R_space'], self.ParamSpace['H_space']]
        State_space = pd.MultiIndex.from_product(iterables)

        Q_space = pd.DataFrame(index = State_space, columns = self.ParamSpace['A_space']).fillna(0)
        Q_space.index.set_names(['Return','Holding'],inplace=True)
        Q_space.columns.set_names(['Action'],inplace=True)
        # initialize the Qvalues for action 0 as slightly greater than 0 so that
        # 'doing nothing' becomes the default action, instead the default action to be the first column of
        # the dataframe.
        Q_space[0] = 0.0000000001

        self.Q_space = Q_space

    # Q TABLE FUNCTIONS

    def getQvalue(self,state):

        ''' 
        Get the set of action available in that state 
        '''

        ret = state[0]
        holding = state[1]
        return self.Q_space.loc[(ret, holding),]

    def argmaxQ(self,state):

        ''' 
        Get the action that maximizes reward in that state 
        '''

        return self.getQvalue(state).idxmax()

    def getMaxQ(self,state):

        ''' 
        Get the action-value function for the action that maximizes reward in that state 
        '''

        return self.getQvalue(state).max()


    def chooseAction(self,state):

        ''' 
        Pick the action greedily 
        '''

        random_action = np.random.rand()
        if (random_action < self.Param['epsilon']):
            # pick one action at random for exploration purposes
            A_space = self.ParamSpace['A_space'] 
            dn = A_space[np.random.randint(len(A_space))]
        else:
            # pick the greedy action
            dn = self.argmaxQ(state)

        return dn


    # FUNCTION FOR COMPUTING OPTIMAL SOLUTION OF GARLEANU-PEDERSEN
    
    def OptTradingRate(self):
    
        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-self.Param['DiscountRate']/260)  
        
        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = (self.Param['kappa']* ( 1 - rho) + self.Param['CostMultiplier']*rho)
        num2 = np.sqrt(num1**2 + 4 *self.Param['kappa'] * self.Param['CostMultiplier']* (1 - rho)**2)
        den = 2* (1 - rho)
        a = (-num1 + num2)/ den
        
        OptRate = a /self.Param['CostMultiplier']
    
        return OptRate
    
    def ChooseOptimalHolding(self, OptCurrHolding, CurrFactors, CurrRet, OptRate):

        '''
        Compute the optimal reward by the analytical solution of Garleanu Pedersen
        '''
                
        if self.Param['NumFactors'] == 1:
            
            # Computed discounted factor loadings according to their speed of mean reversion and 
            # the optimal trading rate
            # we could also take out this computation fromt the function because the discfactor depends on fixed params
            DiscFactorLoads = self.Param['f_param'][0]/ (1 + self.f_speed * ((OptRate * self.Param['CostMultiplier']) / \
                                                                             self.Param['kappa']))
                
            # Optimal traded quantity between period
            OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * \
                          (1/(self.Param['kappa'] * (self.Param['sigma'])**2)) * \
                           (DiscFactorLoads * CurrFactors)
               
            #pdb.set_trace()             
            # Traded quantity as for the Markovitz framework  (Mean-Variance framework)            
            MVNextHolding =  (1/(self.Param['kappa'] * (self.Param['sigma'])**2)) * \
                           (self.Param['f_param'][0] * CurrFactors)
                       
        else:
            # Computed discounted factor loadings according to their speed of mean reversion and 
            # the optimal trading rate
            DiscFactorLoads = self.Param['f_param']/ (1 + self.f_speed * OptRate)
                
            # Optimal traded quantity between period
            OptNextHolding = (1 - OptRate) * OptCurrHolding + OptRate * \
                          (1/(self.Param['kappa'] * (self.Param['sigma'])**2)) * \
                           (DiscFactorLoads @ CurrFactors)
               
                           
            # Traded quantity as for the Markovitz framework  (Mean-Variance framework)            
            MVNextHolding =  (1/(self.Param['kappa'] * (self.Param['sigma'])**2)) * \
                           (self.Param['f_param'] @ CurrFactors)

        # Traded quantity between period
        OptNextAction = OptNextHolding - OptCurrHolding
        # Portfolio variation
        OptGrossPNL = OptNextHolding * CurrRet #(DiscFactorLoads * CurrFactors)
        # Risk
        OptRisk = 0.5 * self.Param['kappa'] * ((OptNextHolding)**2 * (self.Param['sigma'])**2)
        # Transaction costs
        OptCost = self.TotalCost(OptNextHolding - OptCurrHolding)
        # Portfolio Variation including costs
        OptNetPNL = OptGrossPNL - OptCost
        # Compute reward    
        OptReward = OptGrossPNL - OptRisk - OptCost
        
        # Store quantities
        Result = {
                  'OptNextAction': OptNextAction,
                  'OptCurrHolding': OptCurrHolding,
                  'OptNextHolding': OptNextHolding,
                  'OptGrossPNL': OptGrossPNL,
                  'OptNetPNL': OptNetPNL,
                  'OptRisk': OptRisk,
                  'OptCost': OptCost,
                  'OptReward' : OptReward,
                  'MVNextHolding' : MVNextHolding
                  }
        return Result
    
    # TRAIN FUNCTION
    
    def QLearning(self, seed=None):

        '''
        QLearning algorithm
        '''
        
        # pick samples
        realret, ret, factors = self.ReturnSampler(self.Param['N_train'] , seed)
        
        #pdb.set_trace()
        
        res_df = pd.DataFrame(np.nan, 
                              columns = ['returns','factors'], 
                              index = np.arange(len(ret)))
        
        res_df['returns'] = ret
        res_df['factors'] = factors
        
        # initialize holding at first holding possible
        currHolding = self.find_nearest_holding(0)
        
        for i in tqdm(iterable=range(0, self.Param['N_train']+1), desc='Training QLearning'):
            
            #pdb.set_trace()
            # indexing
            currRet = ret[i]
            currState = (currRet, currHolding)

            # choose action
            shares_traded = self.chooseAction(currState)
            nextHolding = self.find_nearest_holding(currHolding + shares_traded)
            nextRet = ret[i+1]
            nextState = (nextRet, nextHolding)

            Result = self.GetReward(currState, nextState)
            q_sa = self.Q_space.loc[currState, shares_traded]
            increment = self.Param['alpha'] * ( Result['Reward'] + \
                                  self.Param['gamma'] * self.getMaxQ(nextState) - q_sa)
            self.Q_space.loc[currState, shares_traded] = q_sa + increment
            
            if i==0:
                for key in Result.keys(): 
                    res_df[key] = 0                 
                    res_df[key].iloc[i] = Result[key]
            else:
                for key in Result.keys():
                    res_df[key].iloc[i] = Result[key]  

            currHolding = nextHolding
            
            if self.Param['plot_insample']:
                if (i % (self.Param['N_train']/5) == 0) & (i != 0):
                    self.PlotLearningResults(res_df.loc[:i], title='In-sample Plots',
                                             plot_GP=0, iteration=i)


    # TEST FUNCTION
    def OutOfSample(self, seed=None):

        '''
        Simulate another price series and test the Q-function learned, while testing also 
        the analytical formula of Garleanu and Pedersen
        '''
        # Sample factors, returns and price for the test
        realret, ret, factors = self.ReturnSampler(self.Param['TestSteps'],seed)
        
        # Initialize dataframe for storing results
        res_df = pd.DataFrame(np.nan, 
                              columns = ['returns','factors'], 
                              index = np.arange(len(ret)))
        
        res_df['returns'] = ret
        res_df['realreturns'] = realret
        res_df['factors'] = factors
        
        # Initialize first holding and optimal holding as same value
        currHolding = self.find_nearest_holding(0)
        curroptHolding = self.find_nearest_holding(0)
        
        # Compute optimal trading rate for analytical solution
        OptRate = self.OptTradingRate()
 
        for i in tqdm(iterable=range(0, self.Param['TestSteps'] + 1),desc='Testing QLearning'):
                        
            # Qlearning out-of-sample
            currRet = ret[i]
            currState = (currRet, currHolding)
            shares_traded = self.chooseAction(currState)

            nextHolding = self.find_nearest_holding(currHolding + shares_traded)

            nextRet = ret[i+1]
            nextState = (nextRet, nextHolding)

            Result = self.GetReward(currState, nextState)
                
            if i==0:
                for key in Result.keys(): 
                    res_df[key] = 0                 
                    res_df[key].iloc[i] = Result[key]
            else:
                for key in Result.keys():
                    res_df[key].iloc[i] = Result[key]      

            currHolding = nextHolding
                     
            # Analytical solutions
            currFactors = factors[i]
            currReturn = realret[i]
    
            OptResult = self.ChooseOptimalHolding(curroptHolding, currFactors, currReturn, OptRate)
                
            if i==0:
                for key in OptResult.keys(): 
                    res_df[key] = 0                 
                    res_df[key].iloc[i] = OptResult[key]
            else:
                for key in OptResult.keys():
                    res_df[key].iloc[i] = OptResult[key] 
    
            curroptHolding = OptResult['OptNextHolding']
            
            #pdb.set_trace()
            
        return res_df
    
    # LAUNCH EXPERIMENT FUNCTION
    def TrainTestQTrader(self):
    
        '''
        Function to train and test the QTrader according to the parameter dict provided
        '''
        
        # Create directory for outputs
        
        savepath = os.path.join(os.getcwd(),
                                'outputs',
                                self.Param['outputDir'],
                                self.Param['outputName'])
        
        if not os.path.exists(savepath):
            os.mkdir(savepath)
            
        # store parameters configuration
        SaveConfig(self.Param,savepath)
        
        # Initialize QTable
        self.CreateQTable()

        # TRAIN
        self.QLearning()

        # TEST
        # fixed seed for reproducibility
        res_df = self.OutOfSample(seed=self.Param['Seed'])
        #res_df[['OptNextHolding','MVNextHolding']].plot()
        
        # plot results including analytical solution     
        self.PlotLearningResults(res_df, title='Out-of-sample Plots', 
                                 plot_GP=1, iteration = self.Param['TestSteps'])
        # plot results without analytical solution
        self.PlotLearningResults(res_df, title='Out-of-sample Plots', 
                                 plot_GP=0, iteration = self.Param['TestSteps'])
        


    # # FUNCTION WRAPPER
    # def MCRun(self):
        
    #     # for _ in range(self.Param['iteration']):
        
    #     #     self.TrainTestQTrader()
        
    #     # Create directory for outputs
    #     savepath = os.path.join(os.getcwd(),
    #                             'outputs',
    #                             self.Param['outputDir'],
    #                             self.Param['outputName'])
        
    #     if not os.path.exists(savepath):
    #         os.mkdir(savepath)
            
    #     # store parameters configuration
    #     SaveConfig(self.Param,savepath)
        


    # MONITORING FUNCTIONS
    def PlotLearningResults(self,res_df, title, plot_GP, iteration):
        
        #if title=='Out-of-sample Plots':
        if plot_GP:
            
            ############################################################################
            # first figure GP
            fig = plt.figure(figsize=(34,13))
            fig.tight_layout()
            plt.suptitle(title,fontsize=28)
    
            # first plot
            axpnl = fig.add_subplot(2,2,1)
            axpnl.plot(res_df[['GrossPNL','OptGrossPNL']].cumsum())
            axpnl.set_title('GrossPNL')
            axpnl.legend(['GrossPNL','OptGrossPNL'])
            GrossPNLmean, GrossPNLstd, GrossPNLsr = self.ComputeSharpeRatio(res_df['GrossPNL'])
            OptGrossPNLmean, OptGrossPNLstd, OptGrossPNLsr = self.ComputeSharpeRatio(res_df['OptGrossPNL'])
            grosspnl_text = AnchoredText(' Gross Sharpe Ratio: ' + str(np.around(GrossPNLsr,2)) + 
                                          '\n Gross PnL mean: ' + str(np.around(GrossPNLmean,2)) + 
                                          '\n Gross PnL std: ' + str(np.around(GrossPNLstd,2)) +
                                          '\n OptGross Sharpe Ratio: ' + str(np.around(OptGrossPNLsr,2)) + 
                                          '\n OptGross PnL mean: ' + str(np.around(OptGrossPNLmean,2)) + 
                                          '\n OptGross PnL std: ' + str(np.around(OptGrossPNLstd,2)), 
                                          loc=4, prop=dict(size=10)) # pad=0, borderpad=0, frameon=False 

            axpnl.add_artist(grosspnl_text)
            
            #second plot
            axnetpnl = fig.add_subplot(2,2,2)
            axnetpnl.plot(res_df[['NetPNL','OptNetPNL']].cumsum())
            axnetpnl.set_title('NetPNL')
            axnetpnl.legend(['NetPNL','OptNetPNL'])
            NetPNLmean, NetPNLstd, NetPNLsr = self.ComputeSharpeRatio(res_df['NetPNL'])
            OptNetPNLmean, OptNetPNLstd, OptNetPNLsr = self.ComputeSharpeRatio(res_df['OptNetPNL'])
            netpnl_text = AnchoredText(' Net Sharpe Ratio: ' + str(np.around(NetPNLsr,2)) + 
                                          '\n Net PnL mean: ' + str(np.around(NetPNLmean,2)) + 
                                          '\n Net PnL std: ' + str(np.around(NetPNLstd,2)) +
                                          '\n OptNet Sharpe Ratio: ' + str(np.around(OptNetPNLsr,2)) + 
                                          '\n OptNet PnL mean: ' + str(np.around(OptNetPNLmean,2)) + 
                                          '\n OptNet PnL std: ' + str(np.around(OptNetPNLstd,2)), 
                                          loc=4,  prop=dict(size=10) )
            axnetpnl.add_artist(netpnl_text)
            
            #third plot
            axreward = fig.add_subplot(2,2,3)
            axreward.plot(res_df[['Reward','OptReward']].cumsum())
            axreward.set_title('Reward')
            axreward.legend(['Reward','OptReward'])
            
            #fourth plot
            axcumcost = fig.add_subplot(2,2,4)
            axcumcost.plot(res_df[['Cost','OptCost']].cumsum())
            axcumcost.set_title('Cumulative Cost')
            axcumcost.legend(['CumulativeCost','CumulativeOptCost'])
            
            
            fold = os.path.join('outputs',self.Param['outputDir'],self.Param['outputName'])
            
            
            figpath = os.path.join(fold,'GP_Qlearning_cumplot_'+ 
                                    format_tousands(self.Param['N_train']) + '_' +
                                    title + '_iteration_' + str(iteration) + '.PNG')
            
            # save figure
            plt.savefig(figpath)
            
            ###############################################################################
            #second figure GP
            fig2 = plt.figure(figsize=(34,13))
            fig2.tight_layout()
            plt.suptitle(title,fontsize=28)
                
            # first plot
            axcost = fig2.add_subplot(4,1,1)
            axcost.plot(res_df[['Cost','OptCost']])
            #axcost.set_title('Cost')
            axcost.legend(['Cost','OptCost'])
                                 
            # second plot
            axrisk = fig2.add_subplot(4,1,2)
            axrisk.plot(res_df[['Risk','OptRisk']])
            #axrisk.set_title('Risk')
            axrisk.legend(['Risk','OptRisk'])

            # third plot
            axaction = fig2.add_subplot(4,1,3)
            axaction.plot(res_df[['Action','OptNextAction']])
            #axaction.set_title('Action')
            axaction.legend(['Action','OptNextAction'])
            
            # fourth plot
            axholding = fig2.add_subplot(4,1,4)
            axholding.plot(res_df[['NextHolding','OptNextHolding']])
            #axholding.set_title('Holding')
            axholding.legend(['NextHolding','OptNextHolding'])
            
            
            figpath = os.path.join(fold,
                                   'GP_Qlearning_plot'+ 
                                    format_tousands(self.Param['N_train']) + '_' +
                                    title + '_iteration_' + str(iteration) + '.PNG')
            
            # save figure
            plt.savefig(figpath)
        
        else:
            
            ############################################################################
            # first figure Ritter
            fig = plt.figure(figsize=(34,13))
            fig.tight_layout()
            plt.suptitle(title,fontsize=28)
    
            
            # first plot
            axpnl = fig.add_subplot(2,2,1)
            axpnl.plot(res_df['GrossPNL'].cumsum())
            axpnl.set_title('GrossPNL')
            axpnl.legend(['GrossPNL'])
            GrossPNLmean, GrossPNLstd, GrossPNLsr = self.ComputeSharpeRatio(res_df['GrossPNL'])
            grosspnl_text = AnchoredText(' Gross Sharpe Ratio: ' + str(np.around(GrossPNLsr,2)) + 
                                          '\n Gross PnL mean: ' + str(np.around(GrossPNLmean,2)) + 
                                          '\n Gross PnL std: ' + str(np.around(GrossPNLstd,2)), 
                                          loc=4, prop=dict(size=10)  )
            axpnl.add_artist(grosspnl_text)
            
            # second plot
            axnetpnl = fig.add_subplot(2,2,2)
            axnetpnl.plot(res_df['NetPNL'].cumsum())
            axnetpnl.set_title('NetPNL')
            axnetpnl.legend(['NetPNL'])
            NetPNLmean, NetPNLstd, NetPNLsr = self.ComputeSharpeRatio(res_df['NetPNL'])
            netpnl_text = AnchoredText(' Net Sharpe Ratio: ' + str(np.around(NetPNLsr,2)) + 
                                          '\n Net PnL mean: ' + str(np.around(NetPNLmean,2)) + 
                                          '\n Net PnL std: ' + str(np.around(NetPNLstd,2)), 
                                          loc=4, prop=dict(size=10) )
            axnetpnl.add_artist(netpnl_text)


            # third plot 
            axreward = fig.add_subplot(2,2,3)
            axreward.plot(res_df['Reward'].cumsum())
            axreward.set_title('Reward')
            axreward.legend(['Reward'])
            
            # fourth plot
            axcumcost = fig.add_subplot(2,2,4)
            axcumcost.plot(res_df['Cost'].cumsum())
            axcumcost.set_title('CumCost')
            axcumcost.legend(['CumCost'])
            
            
            fold = os.path.join(os.getcwd(),'outputs',self.Param['outputDir'],self.Param['outputName'])
            subfold = 'In_sample_Plots_' + format_tousands(self.Param['N_train'])
            
            if title == 'In-sample Plots':
                
                if not os.path.exists(os.path.join(fold,subfold)):
                    os.mkdir(os.path.join(fold,subfold))
                
                figpath = os.path.join(fold,subfold,
                                       'Ritter_Qlearning_cumplot_'+ 
                                        format_tousands(self.Param['N_train']) + '_' +
                                        title + '_iteration_' + str(iteration) + '.PNG')
            else:
                
                figpath = os.path.join(fold,
                                       'Ritter_Qlearning_cumplot_'+ 
                                        format_tousands(self.Param['N_train']) + '_' +
                                        title + '_iteration_' + str(iteration) + '.PNG')
            # save figure
            plt.savefig(figpath)
            
            ############################################################################
            # second figure Ritter
            fig2 = plt.figure(figsize=(34,13))
            fig2.tight_layout()
            plt.suptitle(title,fontsize=28)
            
            # first figure
            axcost = fig2.add_subplot(4,1,1)
            axcost.plot(res_df['Cost'])
            #axcost.set_title('Cost')
            axcost.legend(['Cost'])
                                 
            # second plot
            axrisk = fig2.add_subplot(4,1,2)
            axrisk.plot(res_df['Risk'])
            #axrisk.set_title('Risk')
            axrisk.legend(['Risk'])

            # third plot
            axaction = fig2.add_subplot(4,1,3)
            axaction.plot(res_df['Action'])
            #axaction.set_title('Action')
            axaction.legend(['Action'])
            
            # fourth plot
            axholding = fig2.add_subplot(4,1,4)
            axholding.plot(res_df['NextHolding'])
            #axholding.set_title('Holding')
            axholding.legend(['NextHolding'])
            
            if title == 'In-sample Plots':
           
                figpath = os.path.join(fold,subfold,
                                       'Ritter_Qlearning_plot_'+ 
                                        format_tousands(self.Param['N_train']) + '_' +
                                        title + '_iteration_' + str(iteration) + '.PNG')
            else:
                
                figpath = os.path.join(fold,
                                       'Ritter_Qlearning_plot_'+ 
                                        format_tousands(self.Param['N_train']) + '_' +
                                        title + '_iteration_' + str(iteration) + '.PNG')

            # save figure
            plt.savefig(figpath)

            
    # GENERAL TOOLS
    def ComputeSharpeRatio(self,series):
        
        mean = np.array(series).mean()
        std = np.array(series).std()
        sr = (mean/std) * (252 ** 0.5)
        
        return mean, std, sr