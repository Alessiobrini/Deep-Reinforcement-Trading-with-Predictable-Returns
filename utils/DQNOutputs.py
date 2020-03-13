# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:14:30 2020

@author: aless
"""
import os
from typing import Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from utils.format_tousands import format_tousands
import seaborn 
seaborn.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['savefig.dpi'] = 90
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

import matplotlib
matplotlib.use('Agg')


    # # MONITORING FUNCTIONS
def PlotLearningResults(res_df, title, iteration, executeRL, savedpath, N_train, plot_GP=0):
    
    '''
    Generate intermediate plot of the results during runtime or in aggregate
    with respect to the optimal benchmark after the training.
    
    Parameters
    ----------
    res_df: pandas dataframe
        Dataframe containing time series of results to plot
    
    title: str
        Title selected for the plot
    
    iteration: int
        Number of iteration to distinguish different steps in the learning process
    
    plot_GP=0 : bool
        Boolean to decide if plot thelearning results wrt the GP benchmark.
        Default is not plotting it.
    
    Returns
    -------
    Produce and save plot
    '''

   
    if plot_GP:
        
        ############################################################################
        # first figure GP
        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle(title,fontsize=28)

        # first plot
        axpnl = fig.add_subplot(2,2,1)
        axpnl.plot(res_df['OptGrossPNL'].cumsum(), label = 'OptGrossPNL', color = 'orange')
        axpnl.set_title('GrossPNL')
        
        #second plot
        axnetpnl = fig.add_subplot(2,2,2)
        axnetpnl.plot(res_df['OptNetPNL'].cumsum(), label = 'OptNetPNL', color = 'orange')
        axnetpnl.set_title('NetPNL')
        
        #third plot
        axreward = fig.add_subplot(2,2,3)
        axreward.plot(res_df['OptReward'].cumsum(), label = 'OptReward', color = 'orange')
        axreward.set_title('Reward')
        
        
        #fourth plot
        axcumcost = fig.add_subplot(2,2,4)
        axcumcost.plot(res_df['OptCost'].cumsum(), label = 'OptCost', color = 'orange')
        axcumcost.set_title('Cumulative Cost')
        
        
        if executeRL:
            axpnl.plot(res_df['GrossPNL'].cumsum(), label = 'GrossPNL', color = 'blue')
            
            GrossPNLmean, GrossPNLstd, GrossPNLsr = ComputeSharpeRatio(res_df['GrossPNL'])
            OptGrossPNLmean, OptGrossPNLstd, OptGrossPNLsr = ComputeSharpeRatio(res_df['OptGrossPNL'])
            grosspnl_text = AnchoredText(' Gross Sharpe Ratio: ' + str(np.around(GrossPNLsr,2)) + 
                                          '\n Gross PnL mean: ' + str(np.around(GrossPNLmean,2)) + 
                                          '\n Gross PnL std: ' + str(np.around(GrossPNLstd,2)) +
                                          '\n OptGross Sharpe Ratio: ' + str(np.around(OptGrossPNLsr,2)) + 
                                          '\n OptGross PnL mean: ' + str(np.around(OptGrossPNLmean,2)) + 
                                          '\n OptGross PnL std: ' + str(np.around(OptGrossPNLstd,2)), 
                                          loc=4, prop=dict(size=10)) # pad=0, borderpad=0, frameon=False 

            axpnl.add_artist(grosspnl_text)
            
            
            axnetpnl.plot(res_df['NetPNL'].cumsum(), label = 'NetPNL', color = 'blue')
            
            NetPNLmean, NetPNLstd, NetPNLsr = ComputeSharpeRatio(res_df['NetPNL'])
            OptNetPNLmean, OptNetPNLstd, OptNetPNLsr = ComputeSharpeRatio(res_df['OptNetPNL'])
            netpnl_text = AnchoredText(' Net Sharpe Ratio: ' + str(np.around(NetPNLsr,2)) + 
                                          '\n Net PnL mean: ' + str(np.around(NetPNLmean,2)) + 
                                          '\n Net PnL std: ' + str(np.around(NetPNLstd,2)) +
                                          '\n OptNet Sharpe Ratio: ' + str(np.around(OptNetPNLsr,2)) + 
                                          '\n OptNet PnL mean: ' + str(np.around(OptNetPNLmean,2)) + 
                                          '\n OptNet PnL std: ' + str(np.around(OptNetPNLstd,2)), 
                                          loc=4,  prop=dict(size=10) )
            axnetpnl.add_artist(netpnl_text)
            
            axreward.plot(res_df['Reward'].cumsum(), label = 'Reward', color = 'blue')
            axcumcost.plot(res_df['Cost'].cumsum(), label = 'Cost', color = 'blue')
        else:
            
            OptGrossPNLmean, OptGrossPNLstd, OptGrossPNLsr = ComputeSharpeRatio(res_df['OptGrossPNL'])
            grosspnl_text = AnchoredText(' OptGross Sharpe Ratio: ' + str(np.around(OptGrossPNLsr,2)) + 
                                          '\n OptGross PnL mean: ' + str(np.around(OptGrossPNLmean,2)) + 
                                          '\n OptGross PnL std: ' + str(np.around(OptGrossPNLstd,2)), 
                                          loc=4, prop=dict(size=10)) # pad=0, borderpad=0, frameon=False 

            axpnl.add_artist(grosspnl_text)
            
            OptNetPNLmean, OptNetPNLstd, OptNetPNLsr = ComputeSharpeRatio(res_df['OptNetPNL'])
            netpnl_text = AnchoredText(' OptNet Sharpe Ratio: ' + str(np.around(OptNetPNLsr,2)) + 
                                          '\n OptNet PnL mean: ' + str(np.around(OptNetPNLmean,2)) + 
                                          '\n OptNet PnL std: ' + str(np.around(OptNetPNLstd,2)), 
                                          loc=4,  prop=dict(size=10) )
            axnetpnl.add_artist(netpnl_text)
         

        axpnl.legend() 
        axnetpnl.legend()
        axreward.legend()
        axcumcost.legend()
        
        figpath = os.path.join(savedpath,'GP_Qlearning_cumplot_'+ 
                                format_tousands(N_train) +'.PNG')
        
        # save figure
        plt.savefig(figpath)
        plt.close('all')
        
        ###############################################################################
        #second figure GP
        fig2 = plt.figure(figsize=(34,13))
        fig2.tight_layout()
        plt.suptitle(title,fontsize=28)
            
        # first plot
        axcost = fig2.add_subplot(4,1,1)
        axcost.plot(res_df['OptCost'], label = 'OptCost', alpha=0.7, color = 'orange') 
        axcost.set_title('Cost', x=-0.1,y=0.5)     
        # second plot
        axrisk = fig2.add_subplot(4,1,2)
        axrisk.plot(res_df['OptRisk'], label = 'OptRisk', alpha=0.7, color = 'orange')
        axrisk.set_title('Risk', x=-0.1,y=0.5)
        # third plot
        axaction = fig2.add_subplot(4,1,3)
        axaction.plot(res_df['OptNextAction'], label = 'OptNextAction', alpha=0.7, color = 'orange')
        axaction.set_title('Action', x=-0.1,y=0.5)
        # fourth plot
        axholding = fig2.add_subplot(4,1,4)
        axholding.plot(res_df['OptNextHolding'],label = 'OptNextHolding', alpha=0.7, color = 'orange')
        axholding.set_title('Holding', x=-0.1,y=0.5)
        
        if executeRL:
            axcost.plot(res_df['Cost'], label = 'Cost', color = 'blue')
            axrisk.plot(res_df['Risk'], label = 'Risk', color = 'blue')
            axaction.plot(res_df['Action'], label = 'Action', color = 'blue')
            axholding.plot(res_df['NextHolding'],label = 'NextHolding', color = 'blue')


        axcost.legend()
        axrisk.legend()
        axaction.legend()
        axholding.legend()
        
        figpath = os.path.join(savedpath,'GP_Qlearning_plot_'+
                                format_tousands(N_train) +'.PNG')
        
        # save figure
        plt.savefig(figpath)
        plt.close('all')
    else:
        
        ############################################################################
        # first figure Ritter
        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle(title,fontsize=28)

        
        # first plot
        axpnl = fig.add_subplot(2,2,1)
        axpnl.plot(res_df['GrossPNL'].cumsum(), color = 'blue')
        axpnl.set_title('GrossPNL')
        axpnl.legend(['GrossPNL'])
        GrossPNLmean, GrossPNLstd, GrossPNLsr = ComputeSharpeRatio(res_df['GrossPNL'])
        grosspnl_text = AnchoredText(' Gross Sharpe Ratio: ' + str(np.around(GrossPNLsr,2)) + 
                                      '\n Gross PnL mean: ' + str(np.around(GrossPNLmean,2)) + 
                                      '\n Gross PnL std: ' + str(np.around(GrossPNLstd,2)), 
                                      loc=4, prop=dict(size=10)  )
        axpnl.add_artist(grosspnl_text)
        
        # second plot
        axnetpnl = fig.add_subplot(2,2,2)
        axnetpnl.plot(res_df['NetPNL'].cumsum(), color = 'blue')
        axnetpnl.set_title('NetPNL')
        axnetpnl.legend(['NetPNL'])
        NetPNLmean, NetPNLstd, NetPNLsr = ComputeSharpeRatio(res_df['NetPNL'])
        netpnl_text = AnchoredText(' Net Sharpe Ratio: ' + str(np.around(NetPNLsr,2)) + 
                                      '\n Net PnL mean: ' + str(np.around(NetPNLmean,2)) + 
                                      '\n Net PnL std: ' + str(np.around(NetPNLstd,2)), 
                                      loc=4, prop=dict(size=10) )
        axnetpnl.add_artist(netpnl_text)


        # third plot 
        axreward = fig.add_subplot(2,2,3)
        axreward.plot(res_df['Reward'].cumsum(), color = 'blue')
        axreward.set_title('Reward')
        axreward.legend(['Reward'])
        
        # fourth plot
        axcumcost = fig.add_subplot(2,2,4)
        axcumcost.plot(res_df['Cost'].cumsum(), color = 'blue')
        axcumcost.set_title('CumCost')
        axcumcost.legend(['CumCost'])
        
        figpath = os.path.join(savedpath,'Qlearning_cumplot_'+ 
                                '_iteration_' + str(iteration) + '_' + 
                                format_tousands(N_train) +'.PNG')
        # save figure
        plt.savefig(figpath)
        plt.close('all')
        
        ############################################################################
        # second figure Ritter
        fig2 = plt.figure(figsize=(34,13))
        fig2.tight_layout()
        plt.suptitle(title,fontsize=28)
        
        # first figure
        axcost = fig2.add_subplot(4,1,1)
        axcost.plot(res_df['Cost'], color = 'blue')
        axcost.set_title('Cost', x=-0.1,y=0.5)
        axcost.legend(['Cost'])
                             
        # second plot
        axrisk = fig2.add_subplot(4,1,2)
        axrisk.plot(res_df['Risk'], color = 'blue')
        axrisk.set_title('Risk', x=-0.1,y=0.5)
        axrisk.legend(['Risk'])

        # third plot
        axaction = fig2.add_subplot(4,1,3)
        axaction.plot(res_df['Action'], color = 'blue')
        axaction.set_title('Action', x=-0.1,y=0.5)
        axaction.legend(['Action'])
        
        # fourth plot
        axholding = fig2.add_subplot(4,1,4)
        axholding.plot(res_df['NextHolding'], color = 'blue')
        axholding.set_title('Holding', x=-0.1,y=0.5)
        axholding.legend(['NextHolding'])
                    
        figpath = os.path.join(savedpath,'Qlearning_plot_'+ 
                                '_iteration_' + str(iteration) + '_' + 
                                format_tousands(N_train) +'.PNG')

        # save figure
        plt.savefig(figpath)
        plt.close('all')
            
            

         
    # GENERAL TOOLS
def ComputeSharpeRatio(series: Union[pd.Series or pd.DataFrame]) -> Tuple[float,
                                                                              float,
                                                                              float]:
    
    '''
    Compute SharpeRatio measure from a pnl series
    
    Parameters
    ----------
    series : pandas series or dataframe
        series of expected returns (pnl)
    
    Returns
    -------
    mean, standard deviation and Sharpe Ratio of the series
    '''
    
    mean = np.array(series).mean()
    std = np.array(series).std()
    sr = (mean/std) * (252 ** 0.5)
    
    return mean, std, sr

    

        
        