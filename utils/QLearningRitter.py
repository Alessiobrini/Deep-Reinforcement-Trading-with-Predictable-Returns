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

import pickle
import os
import pdb

from tqdm import tqdm

from utils.format_tousands import format_tousands
from utils.Base import Base

class QTraderObject(Base):
    
    '''
    This class inherits from QLearning class.
    It is useful to implement a general Q-learning agent and produce simple 
    synthetic experiments as in Ritter (2017).
    '''


    # PRICE SAMPLERS
    
    def PriceSampler(self, sampleSize, seed=None):
      
        '''
        Simulate a price process according to a selected dynamic (Ornstein-Uhlenbeck)
        '''
        # Set seed to make the out-of-sample experiment reproducible
        np.random.seed(seed)
        # instantiate the random component and the parameters
        eps = np.random.randn(sampleSize)
        pe = self.Param['P_e']
        sigma = self.Param['sigma']
        lambda_ = np.around(np.log(2)/self.Param['HalfLife'],4)
        # started price for the simulation
        p = self.find_nearest_price(pe) 
      
        prices = [p]
        for i in range(0,sampleSize):
            x = np.log(p / pe)
            x = (1 - lambda_) * x + sigma * eps[i] 
            pnew = pe * np.exp(x)
            pnew = np.min([pnew, self.Param['P_max']])
            # discretizing to make sure it appear in P_space
            prices.append(self.find_nearest_price(pnew))
            p = pnew

        return prices


    # COST FUNCTIONS
    
    def TotalCost(self,dn):
        '''
        Calculate total transaction costs 
        '''
        # SpreadCost
        sc = self.Param['TickSize'] * np.abs(dn)
        # ImpactCost
        ic = (dn**2)* self.Param['TickSize']/ self.Param['LotSize']

        return self.Param['CostMultiplier'] * (sc + ic)

    
    # REWARD FUNCTIONS
    
    def GetReward(self, currState, nextState, idx_rm, pnl_df):

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
        # Transaction costs
        cost = self.TotalCost(dn)
        # Price variation
        pdiff = nextPrice - currPrice
        # Portfolio variation
        pnl = nextHolding * pdiff - cost
        # Compute reward    
        if self.Param['RollingMean']:
            if idx_rm < self.Param['LenRollMean']:
                reward = pnl - 0.5 * self.Param['kappa'] * (pnl **2)
            else:
                reward = pnl - 0.5 * self.Param['kappa'] * (pnl - \
                         pnl_df['pnl'].iloc[(idx_rm - self.Param['LenRollMean']) : idx_rm].mean())**2
        else:
            reward = pnl - 0.5 * self.Param['kappa'] * (pnl **2)
        
        # Store quantities
        result = {
                  'currState': currState,
                  'nextState': nextState,
                  'pnl': pnl,
                  'cost': cost,
                  'reward' : reward,
                  'dn' : dn
                  }
        return result

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


    # TRAIN FUNCTION
    
    def QLearning(self, seed=None):

        '''
        QLearning algorithm
        '''

        # index for computing rolling mean of expected returns
        idx_rm = 0
        
        # draw samples
        pricePath = self.PriceSampler(self.Param['N_train'], seed)
        
        # initialize dataframe for keeping track of pnl
        res_df = pd.DataFrame(0, columns = ['pnl','reward','cost','dn'], 
                              index = np.arange(len(pricePath)))

        # initialize holding at first holding possible
        currHolding = self.find_nearest_holding(0)
        
        for i in tqdm(range(0, self.Param['N_train'])):
            # indexing
            currPrice = pricePath[i]
            currState = (currPrice, currHolding)

            # choose action
            shares_traded = self.chooseAction(currState)

            nextHolding = self.find_nearest_holding(currHolding + shares_traded)
            nextPrice = pricePath[i+1]
            nextState = (nextPrice, nextHolding)

            result = self.GetReward(currState, nextState, idx_rm, res_df)
            res_df['pnl'].iloc[i] = result['pnl']
            res_df['reward'].iloc[i] = result['reward']
            res_df['cost'].iloc[i] = result['cost']
            res_df['dn'].iloc[i] = result['dn']
            #train_results['TrainResult' + str(i+1)] = result
            q_sa = self.Q_space.loc[currState, shares_traded]
            increment = self.Param['alpha'] * ( result['reward'] + \
                                  self.Param['gamma'] * self.getMaxQ(nextState) - q_sa)
            self.Q_space.loc[currState, shares_traded] = q_sa + increment

            currHolding = nextHolding
            
            idx_rm += 1
            

        #self.train_results = train_results

    # TEST FUNCTION
    def OutOfSample(self, seed=None):

        '''
        Simulate another price series and test the Q-function learned
        '''
        
        idx_rm = 0
        
        pricePath = self.PriceSampler(self.Param['TestSteps'],seed)
        
        res_df = pd.DataFrame(0, columns = ['pnl','reward','cost','dn'],
                              index = np.arange(len(pricePath)))
        
        currHolding = self.find_nearest_holding(0)
        # empty dict to store results
        test_results = {}
        
        for i in tqdm(range(0, self.Param['TestSteps'])):
            
            currPrice = pricePath[i]
            currState = (currPrice, currHolding)
            shares_traded = self.chooseAction(currState)

            nextHolding = self.find_nearest_holding(currHolding + shares_traded)

            nextPrice = pricePath[i+1]
            nextState = (nextPrice, nextHolding)

            result = self.GetReward(currState, nextState, idx_rm, res_df)
            res_df['pnl'].iloc[i] = result['pnl']
            res_df['reward'].iloc[i] = result['reward']
            res_df['cost'].iloc[i] = result['cost']
            res_df['dn'].iloc[i] = result['dn']

            test_results['TestResult' + str(i+1)] = result

            currHolding = nextHolding
            
            idx_rm += 1

        self.test_results = test_results

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

        # compute Sharpe Ratio of the PNL
        pnl_mean = np.array(res_df['pnl']).mean()
        pnl_std = np.array(res_df['pnl']).std()
        sr = (pnl_mean/pnl_std) * (252 ** 0.5)
        # store test result
        test_results = self.test_results
        # cumulative PNL
        pnl_sum = res_df['pnl'].cumsum()


        # # set up figure and axes
        # fig, ax = plt.subplots(1,1)
        # anchored_text = AnchoredText('Sharpe Ratio: ' + str(np.around(sr,4)) + 
        #                              '\n PnL mean: ' + str(np.around(pnl_mean,2)) + 
        #                              '\n PnL std: ' + str(np.around(pnl_std,2)), loc=4 )
        # ax.plot(pnl_sum)
        # ax.add_artist(anchored_text)
        # plt.xlabel('out−of−sample periods')
        # plt.ylabel('PnL')
        # plt.title('Simulated net PnL over 5000 out−of−sample periods \n' +
        #           format_tousands(self.Param['N_train']) + ' training steps \n K=' +
        #           str(self.Param['K']) + ' M=' + str(self.Param['M']) +
        #           '\n CM=' + str(self.Param['CostMultiplier']))
        
        # # create figure path
        # figpath = os.path.join('outputs', self.Param['outputDir'], 'Qlearning_'+ 
        #                        format_tousands(self.Param['N_train']) +
        #                        '_steps_K=' + str(self.Param['K']) +
        #                        '_M=' + str(self.Param['M']) +
        #                        '_CM=' + str(self.Param['CostMultiplier']) +
        #                        '.PNG')#.replace("/","\\")
        
        # # save figure
        # plt.savefig(figpath)

        # # create table path
        # tablepath = os.path.join('outputs', self.Param['outputDir'], 'QTable_' + 
        #                           format_tousands(self.Param['N_train']) + 
        #                           '_steps_K=' + str(self.Param['K']) + 
        #                           '_M=' + str(self.Param['M']) +
        #                           '_CM=' + str(self.Param['CostMultiplier']) +
        #                           '.csv')#.replace("/","\\")

        # # save qtable as csv
        # self.Q_space.to_csv(tablepath)
        
        
        # # create testresults path
        # testpath = os.path.join('outputs', self.Param['outputDir'], 'test_results_'+ 
        #                         format_tousands(self.Param['N_train']) + 
        #                         '_steps_K=' + str(self.Param['K']) + 
        #                         '_M=' + str(self.Param['M']) +
        #                         '_CM=' + str(self.Param['CostMultiplier']))#.replace("/","\\")

        # # save test results as pickle file
        # with open(testpath,'wb') as filetosave:

        #     pickle.dump(test_results, filetosave)
            
        return res_df
            
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
        