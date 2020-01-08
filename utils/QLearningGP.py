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

import pdb
import pickle
import os
from tqdm import tqdm

from utils.format_tousands import format_tousands
from utils.Base import Base

class QTraderObject(Base):
    
    '''
    This class inherits from QLearning class.
    It is useful to implement a Q-learning agent that produces and compare the simple 
    synthetic experiments as in Ritter (2017) and the Dynamic Programming analytical solution of
    Garleanu and Pedersen (2013).
    '''


    # PRICE SAMPLERS
    
    def PriceSampler(self, sampleSize, seed=None, plot=False):
      
        '''
        Simulate a price process according to a selected dynamic (Ornstein-Uhlenbeck)
        '''
        # Set seed to make the out-of-sample experiment reproducible
        np.random.seed(seed)
        # simulate factors according to OU process
        eps = np.random.randn(sampleSize) 
        sigmaf = self.Param['sigmaf']
        lambda_5D = np.around(np.log(2)/self.Param['HalfLife'][0],4)
        lambda_1Y = np.around(np.log(2)/self.Param['HalfLife'][1],4)
        lambda_5Y = np.around(np.log(2)/self.Param['HalfLife'][2],4)
        
        f0_5D = self.Param['f_init'][0]
        f5D = [f0_5D]
        f0_1Y = self.Param['f_init'][1] 
        f1Y = [f0_1Y]
        f0_5Y = self.Param['f_init'][2] 
        f5Y = [f0_5Y]
        
        # suppose that they are correctly simulated (TODO : check starting value)
        for i in tqdm(range(sampleSize - 1)):
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
        
        # calculate return series
        u = np.random.randn(sampleSize)
        sigma = self.Param['sigma']
        ret = [(f_param.T @ factors[i]) + sigma * u[i] for i in range(sampleSize)]
        #pdb.set_trace()
        
        # started price for the simulation
        pe = self.Param['P_e']
        p = self.find_nearest_price(pe) 
        prices = [p]
        for i in tqdm(range(sampleSize-1)):
            pnew = ret[i] + p # eventually we can add the risk free rate also
            pnew = np.min([pnew, self.Param['P_max']])
            # discretizing to make sure it appear in P_space
            prices.append(self.find_nearest_price(pnew))
            p = pnew
        
        # plots for factor, returns and prices
        if self.Param['plot_inputs']:
            fig1 = plt.figure()
            fig2 = plt.figure()
            fig3 = plt.figure()
            
            ax1 = fig1.add_subplot(111)
            ax2 = fig2.add_subplot(111)
            ax3 = fig3.add_subplot(111)
            
                    
            ax1.plot(f5D)
            ax1.plot(f1Y)
            ax1.plot(f5Y)
            ax1.legend(['5D','1Y','5Y'])
            ax1.set_title('Factors')
            ax2.plot(ret)
            ax2.set_title('Returns')
            ax3.plot(prices)
            ax3.set_title('Prices')
        

        self.f_speed = np.array([lambda_5D,lambda_1Y,lambda_5Y])
   
        return prices, ret, factors 


    # COST FUNCTIONS
    
    def TotalCost(self,dn):
        '''
        Calculate quadratic transaction costs 
        '''
        
        # calculate Kyle's Lambda (Garleanu Pedersen 2013)
        Lambda = self.Param['CostMultiplier'] * self.Param['sigma']**2
        quadratic_costs = 0.5 * dn**2 * Lambda
        
        return quadratic_costs

    
    # REWARD FUNCTIONS
    
    def GetReward(self, currState, nextState):

        '''
        Compute the one step reward
        '''
        
        # Remember that a state is a tuple (price, holding)
        currPrice = currState[0]
        nextPrice = nextState[0]
        currHolding = currState[1]
        nextHolding = nextState[1]

        # Traded quantity between period
        dn = nextHolding - currHolding
        # Price variation
        pdiff = nextPrice - currPrice
        # Portfolio variation
        GrossPNL = nextHolding * pdiff
        # Risk
        Risk = 0.5 * self.Param['kappa'] * (nextHolding**2 * self.Param['sigma']**2)
        # Transaction costs
        Cost = self.TotalCost(dn)
        # Portfolio Variation including costs
        NetPNL = GrossPNL - Cost
        # Compute reward    
        Reward = GrossPNL - Risk - Cost
        
        # Store quantities
        Result = {
                  'currHolding': currHolding,
                  'nextHolding': nextHolding,
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
        iterables = [self.ParamSpace['P_space'], self.ParamSpace['H_space']]
        State_space = pd.MultiIndex.from_product(iterables)

        Q_space = pd.DataFrame(index = State_space, columns = self.ParamSpace['A_space']).fillna(0)
        Q_space.index.set_names(['Price','Holding'],inplace=True)
        Q_space.columns.set_names(['Action'],inplace=True)

        self.Q_space = Q_space

    # Q TABLE FUNCTIONS

    def getQvalue(self,state):

        ''' 
        Get the set of action available in that state 
        '''

        price = state[0]
        holding = state[1]
        return self.Q_space.loc[(price, holding),]

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
        
        OptRate = a/ self.Param['CostMultiplier']
    
        return OptRate
    
    def ChooseOptimalHolding(self, OptCurrAction, CurrFactors, CurrReturns, OptRate):

        '''
        Compute the optimal reward by the analytical solution of Garleanu Pedersen
        '''
        
        # Computed discounted factor loadings according to their speed of mean reversion and 
        # the optimal trading rate
        DiscFactorLoads = self.Param['f_param']/ (1 + self.f_speed * OptRate)
        
        # Optimal traded quantity between period
        OptNextAction = (1 - OptRate) * OptCurrAction + OptRate * \
                      (1/(self.Param['kappa'] * (self.Param['sigma'])**2)) * \
                       (DiscFactorLoads @ CurrFactors)
           
        #nextoptHolding = self.find_nearest_holding(nextoptHolding)
                       
        # Traded quantity as for the Markovitz framework  (Mean-Variance framework)            
        MVNextHolding =  (1/(self.Param['kappa'] * (self.Param['sigma'])**2)) * \
                       (self.Param['f_param'] @ CurrFactors)

        # Traded quantity between period
        #dn = OptNextAction - OptCurrAction
        # Portfolio variation
        OptGrossPNL = OptNextAction * CurrReturns
        # Risk
        OptRisk = 0.5 * self.Param['kappa'] * ((OptNextAction)**2 * (self.Param['sigma'])**2)
        # Transaction costs
        OptCost = self.TotalCost(OptNextAction - OptCurrAction)
        # Portfolio Variation including costs
        OptNetPNL = OptGrossPNL - OptCost
        # Compute reward    
        OptReward = OptGrossPNL - OptRisk - OptCost
        
        # Store quantities
        Result = {
                  'OptCurrAction': OptCurrAction,
                  'OptNextAction': OptNextAction,
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
        pricePath, _, _ = self.PriceSampler(self.Param['N_train'] , seed)
        
        # initialize holding at first holding possible
        currHolding = self.find_nearest_holding(0)
        
        for i in tqdm(range(0, self.Param['N_train'] - 1)):
            # indexing
            currPrice = pricePath[i]
            currState = (currPrice, currHolding)

            # choose action
            shares_traded = self.chooseAction(currState)
            nextHolding = self.find_nearest_holding(currHolding + shares_traded)
            nextPrice = pricePath[i+1]
            nextState = (nextPrice, nextHolding)

            result = self.GetReward(currState, nextState)
            q_sa = self.Q_space.loc[currState, shares_traded]
            increment = self.Param['alpha'] * ( result['Reward'] + \
                                  self.Param['gamma'] * self.getMaxQ(nextState) - q_sa)
            self.Q_space.loc[currState, shares_traded] = q_sa + increment

            currHolding = nextHolding


    # TEST FUNCTION
    def OutOfSample(self, seed=None):

        '''
        Simulate another price series and test the Q-function learned, while testing also 
        the analytical formula of Garleanu and Pedersen
        '''
        # Sample factors, returns and price for the test
        pricePath, returns, factors = self.PriceSampler(self.Param['TestSteps'],seed,True)
        
        # Initialize dataframe for storing results
        res_df = pd.DataFrame(0, 
                              columns = ['price','returns','factors'], 
                              index = np.arange(len(pricePath)))
        
        res_df['price'] = pricePath
        res_df['returns'] = returns
        res_df['factors'] = factors
        
        # Initialize first holding and optimal holding as same value
        currHolding = self.find_nearest_holding(0)
        curroptHolding = self.find_nearest_holding(0)
        
        # Compute optimal trading rate for analytical solution
        OptRate = self.OptTradingRate()
        
        for i in tqdm(range(0, self.Param['TestSteps'] - 1)):
                        
            # Qlearning out-of-sample
            currPrice = pricePath[i]
            currState = (currPrice, currHolding)
            shares_traded = self.chooseAction(currState)

            nextHolding = self.find_nearest_holding(currHolding + shares_traded)

            nextPrice = pricePath[i+1]
            nextState = (nextPrice, nextHolding)

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
            currReturns = returns[i]
            
            OptResult = self.ChooseOptimalHolding(curroptHolding, currFactors,
                                                   currReturns, OptRate)
            if i==0:
                for key in OptResult.keys(): 
                    res_df[key] = 0                 
                    res_df[key].iloc[i] = OptResult[key]
            else:
                for key in OptResult.keys():
                    res_df[key].iloc[i] = OptResult[key] 

            curroptHolding = OptResult['OptNextAction']
            
        return res_df
    
    # LAUNCH EXPERIMENT FUNCTION
    def TrainTestQTrader(self):
    
        '''
        Function to train and test the QTrader according to the parameter dict provided
        '''
        
        # Initialize QTable
        self.CreateQTable()

        # TRAIN
        self.QLearning()

        # TEST
        # fixed seed for reproducibility
        res_df = self.OutOfSample(seed=self.Param['Seed'])
        #res_df[['nextMarkHolding','optholding']].plot()
        res_df[['OptNextAction','MVNextHolding']].plot()
        
        #test plot discretizing the optimal actions
        optimal = [self.find_nearest_holding(h) for h in res_df['OptNextAction']]
        # plt.plot(optimal)

        # # compute Sharpe Ratio of the PNL
        # net_pnl_mean = np.array(res_df['net_pnl']).mean()
        # net_pnl_std = np.array(res_df['net_pnl']).std()
        # net_sr = (net_pnl_mean/net_pnl_std) * (252 ** 0.5)
        # # store test result
        # # test_results = self.test_results
        # # cumulative gross and net PNL
        # pnl_sum = res_df['pnl'].cumsum()
        # net_pnl_sum = res_df['net_pnl'].cumsum()
        
        
        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle('Out-of-sample Plots',fontsize=28)

        axpnl = fig.add_subplot(2,3,1)
        axnetpnl = fig.add_subplot(2,3,2)
        axrisk = fig.add_subplot(2,3,3)
        axcost = fig.add_subplot(2,3,4)
        axreward = fig.add_subplot(2,3,5)
        axholding = fig.add_subplot(2,3,6)
            
        axpnl.plot(res_df[['GrossPNL','OptGrossPNL']].cumsum())
        axpnl.set_title('GrossPNL')
        axpnl.legend(['GrossPNL','OptGrossPNL'])
        
        axnetpnl.plot(res_df[['NetPNL','OptNetPNL']].cumsum())
        axnetpnl.set_title('NetPNL')
        axnetpnl.legend(['NetPNL','OptNetPNL'])
        
        axrisk.plot(res_df[['Risk','OptRisk']])
        axrisk.set_title('Risk')
        axrisk.legend(['Risk','OptRisk'])
        
        axcost.plot(res_df[['Cost','OptCost']])
        axcost.set_title('Cost')
        axcost.legend(['Cost','OptCost'])
        
        axreward.plot(res_df[['Reward','OptReward']].cumsum())
        axreward.set_title('Reward')
        axreward.legend(['Reward','OptReward'])
        
        axholding.plot(res_df[['Action','OptNextAction']])
        axholding.plot(optimal)
        axholding.set_title('Action')
        axholding.legend(['Action','OptNextAction','optcap'])
        
        figpath = os.path.join('outputs',self.Param['outputDir'], 'Qlearning_'+ 
                                format_tousands(self.Param['N_train']) +
                                '_lot=' + str(self.Param['LotSize']) + 
                                '.PNG')
        
        # save figure
        plt.savefig(figpath)


        # # set up figure and axes
        # fig, ax = plt.subplots(1,1)
        # anchored_text = AnchoredText('Net Sharpe Ratio: ' + str(np.around(net_sr,4)) + 
        #                              '\n PnL mean: ' + str(np.around(net_pnl_mean,2)) + 
        #                              '\n PnL std: ' + str(np.around(net_pnl_std,2)), loc=4 )
        # ax.plot(pnl_sum)
        # ax.plot(net_pnl_sum)
        # ax.add_artist(anchored_text)
        # plt.legend(['gross','net'])
        # plt.xlabel('out−of−sample periods')
        # plt.ylabel('PnL')
        # plt.title('Simulated gross and net PnL over 5000 out−of−sample periods \n' +
        #           format_tousands(self.Param['N_train']) + ' training steps \n K=' +
        #           str(self.Param['K']) + ' M=' + str(self.Param['M']) +
        #           '\n CM=' + str(self.Param['CostMultiplier']))
        
        # # create figure path
        # figpath = os.path.join('outputs',self.Param['outputDir'], 'Qlearning_'+ 
        #                        format_tousands(self.Param['N_train']) +
        #                        '_steps_K=' + str(self.Param['K']) +
        #                        '_M=' + str(self.Param['M']) +
        #                        '_CM=' + str(self.Param['CostMultiplier']) +
        #                        '.PNG')
        
        # # save figure
        # plt.savefig(figpath)

        # # create table path
        # tablepath = os.path.join('outputs', self.Param['outputDir'], 'QTable_' + 
        #                          format_tousands(self.Param['N_train']) + 
        #                          '_steps_K=' + str(self.Param['K']) + 
        #                          '_M=' + str(self.Param['M']) +
        #                          '_CM=' + str(self.Param['CostMultiplier']) +
        #                          '.csv')
        # # save qtable as csv
        # self.Q_space.to_csv(tablepath)
        
        
        # # create testresults path
        # testpath = os.path.join('outputs', self.Param['outputDir'], 'test_results_'+ 
        #                         format_tousands(self.Param['N_train']) + 
        #                         '_steps_K=' + str(self.Param['K']) + 
        #                         '_M=' + str(self.Param['M']) +
        #                         '_CM=' + str(self.Param['CostMultiplier']))

        # save test results as pickle file
        # with open(testpath,'wb') as filetosave:

        #     pickle.dump(test_results, filetosave)
            
            
            
    # PLOT FUNCTIONS
    def plot_QValueFunction(self):
        
        '''
        This function accepts the QTable and a dict of parameters as arguments to plot the value function 
        with respect to the price
        '''
        
        # create table path
        tablepath = os.path.join(self.Param['outputDir'], 'QTable_' + 
                                 format_tousands(self.Param['N_train']) + 
                                 '_steps_K=' + str(self.Param['K']) + 
                                 '_M=' + str(self.Param['M']) +
                                 '_CM=' + str(self.Param['CostMultiplier']) +
                                 '.csv')
        
        # Read QTable
        QTable = pd.read_csv(tablepath,index_col= [0,1])
    
        # select values for xaxis
        p = QTable.index.get_level_values(0).unique()
    
        if self.Param['Aggregation']:
            Q_price = QTable.groupby(level = [0]).sum()
    
            fig, ax = plt.subplots(1,1)
    
            for i in range(0,len(Q_price.columns)):
                ax.scatter(p,Q_price.iloc[:,i],label=Q_price.columns[i], s=15)
    
            plt.xlabel('Prices')
            plt.ylabel(r'$\hat{q}((holding,p),a)$',rotation=0, labelpad=30)
            plt.title('Tabular QLearning Value Function by aggregated holding \n' +
                       format_tousands(self.Param['N_train']) + 
                       ' training steps K=' + str(self.Param['K']) + 
                       ' M=' + str(self.Param['M']) +
                       ' CM=' + str(self.Param['CostMultiplier']))
    
            plt.legend(loc=0)
    
        else:
            Q_price = QTable[QTable.index.get_level_values('Holding') == self.Param['holding']]
    
    
            fig, ax = plt.subplots(1,1)
    
            for i in range(0,len(Q_price.columns)):
                ax.scatter(p,Q_price.iloc[:,i],label=Q_price.columns[i], s=15)
    
            plt.xlabel('Prices')
            plt.ylabel(r'$\hat{q}((holding,p),a)$',rotation=0, labelpad=30)
            plt.title('Tabular QLearning Value Function for holding ' +
                      str(self.Param['holding']) + '\n' +
                       format_tousands(self.Param['N_train']) + 
                       ' training steps K=' + str(self.Param['K']) + 
                       ' M=' + str(self.Param['M']) +
                       ' CM=' + str(self.Param['CostMultiplier']))
    
            plt.legend(loc=0)
        
        
    def plot_Actions(self):
    
        '''
        This function accepts the QTable and a dict of parameters as arguments to plot the best action to take 
        with respect to the price and holding
        '''
        
        # create table path
        tablepath = os.path.join(self.Param['outputDir'], 'QTable_' + 
                                 format_tousands(self.Param['N_train']) + 
                                 '_steps_K=' + str(self.Param['K']) + 
                                 '_M=' + str(self.Param['M']) +
                                 '_CM=' + str(self.Param['CostMultiplier']) +
                                 '.csv')
        
        # Read QTable
        QTable = pd.read_csv(tablepath,index_col= [0,1])
       
        if self.Param['Aggregation']:
            Q_price = QTable.groupby(level = [0]).sum()
            Q_action = Q_price.idxmax(axis = 1)
    
            fig, ax = plt.subplots(1,1)
            ax.plot(Q_action)
    
            plt.xlabel('Prices')
            plt.ylabel('Best Action',rotation=0, labelpad=50)
            plt.title('Aggregated best Action for the current price \n' +
                      format_tousands(self.Param['N_train']) + 
                      ' training steps K=' + str(self.Param['K']) + 
                      ' M=' + str(self.Param['M']) +
                      ' CM=' + str(self.Param['CostMultiplier']))
    
        else:
            Q_price = QTable[QTable.index.get_level_values('Holding') == self.Param['holding']]
            Q_price.index = Q_price.index.droplevel(1)
            Q_action = Q_price.idxmax(axis = 1)
    
    
            fig, ax = plt.subplots(1,1)
            ax.plot(Q_action)
    
            plt.xlabel('Prices')
            plt.ylabel('Best Action',rotation=0, labelpad=50)
            plt.title('Best Action for the current (price,holding) pair \n' +
                      format_tousands(self.Param['N_train']) + 
                      ' training steps K=' + str(self.Param['K']) + 
                      ' M=' + str(self.Param['M']) +
                      ' CM=' + str(self.Param['CostMultiplier']))
            

    def plot_Heatmap(self):
        
        '''
        This function accepts the QTable and a dict of parameters as arguments to plot the heatmap for 
        the best action to take with respect to the price and holding
        '''
        
        # create table path
        tablepath = os.path.join(self.Param['outputDir'], 'QTable_' + 
                                 format_tousands(self.Param['N_train']) + 
                                 '_steps_K=' + str(self.Param['K']) + 
                                 '_M=' + str(self.Param['M']) +
                                 '_CM=' + str(self.Param['CostMultiplier']) +
                                 '.csv')
        
        # Read QTable
        QTable = pd.read_csv(tablepath,index_col= [0,1])
        
        Q_action = QTable.idxmax(axis = 1).unstack()
        # change datatype (found on stackoverflow)
        Q_action = Q_action[Q_action.columns].astype(float)
    
        seaborn.heatmap(Q_action, yticklabels=100)
        plt.title('Heatmap of best action for the current price \n' +
                  format_tousands(self.Param['N_train']) + 
                  ' training steps K=' + str(self.Param['K']) + 
                  ' M=' + str(self.Param['M']) +
                  ' CM=' + str(self.Param['CostMultiplier']))
        