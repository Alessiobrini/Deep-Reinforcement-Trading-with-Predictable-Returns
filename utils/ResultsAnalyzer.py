# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:19:05 2020

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

import os, glob, pdb
from utils.format_tousands import format_tousands
from utils.QLearningGPRet import QTraderObject


class ResultsObject(object):

    # ----------------------------------------------------------------------------
    # Init method      
    # ----------------------------------------------------------------------------
    
    def __init__(self, Param):

        '''
        init method to initialize the class. Parameter inputs are stored 
        as properties of the object.
        '''
        self.Param = Param
        self._generatePath(Param)
        
    # ----------------------------------------------------------------------------
    # Private method      
    # ----------------------------------------------------------------------------
    
    def _generatePath(self,Param):
        
        savedpath = os.path.join(os.getcwd(),
                                 self.Param['outputDir'],
                                 self.Param['outputClass'],
                                 '_'.join([self.Param['outputModel'],self.Param['varying_par']]),
                                 format_tousands(self.Param['N_train']))
        
        self.savedpath = savedpath
        
        print(self.savedpath)
    
    # ----------------------------------------------------------------------------
    # Public method       
    # ----------------------------------------------------------------------------


    def PlotLearningSpeed(self):
        
        '''
        Generate plot of the rate of learning of the algorithm
        after training
        '''

        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle('_'.join([self.Param['outputModel'],self.Param['varying_par']]) 
                               + ' Learning Speeds',fontsize=28)
        
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title(self.Param['Var'][0])
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title(self.Param['Var'][1])
        ax3 = fig.add_subplot(2,2,3)
        ax3.set_title(self.Param['Var'][2])
        ax4 = fig.add_subplot(2,2,4)
        ax4.set_title(self.Param['Var'][3])
        

        
        if self.Param['Values'] == 'all':
            
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                       if os.path.isdir(os.path.join(self.savedpath,subdir))]
            
            values = self.Param['Values']
            
        else:
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
            values = [str(val) for val in self.Param['Values']]

    
        for folder in subdirs:
            
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
            
            # first plot
            speed1 = self.ComputeLearningSpeed(res_df,self.Param['Var'][0])
            ax1.plot(speed1.iloc[self.Param['start']:], label=folder.split('_')[-1])
            #second plot
            speed2 = self.ComputeLearningSpeed(res_df,self.Param['Var'][1])
            ax2.plot(speed2.iloc[self.Param['start']:], label=folder.split('_')[-1])
            #third plot
            speed3 = self.ComputeLearningSpeed(res_df,self.Param['Var'][2])
            ax3.plot(speed3.iloc[self.Param['start']:], label=folder.split('_')[-1])
            #fourth plot
            speed4 = self.ComputeLearningSpeed(res_df,self.Param['Var'][3])
            ax4.plot(speed4.iloc[self.Param['start']:], label=folder.split('_')[-1])
            
            if self.Param['plot_speed_post'] == 1:
                
                res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_Test' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
                
                # first plot
                speed1_post = self.ComputeLearningSpeed(res_df_post,self.Param['Var'][0])
                ax1.plot(speed1_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                #second plot
                speed2_post = self.ComputeLearningSpeed(res_df_post,self.Param['Var'][1])
                ax2.plot(speed2_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                #third plot
                speed3_post = self.ComputeLearningSpeed(res_df_post,self.Param['Var'][2])
                ax3.plot(speed3_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                #fourth plot
                speed4_post = self.ComputeLearningSpeed(res_df_post,self.Param['Var'][3])
                ax4.plot(speed4_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                
        
        
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        
        
        
        figpath = os.path.join(self.savedpath,'Qlearning_speeds_'+ 
                                format_tousands(self.Param['N_train']) + '_' +
                                '_'.join(self.Param['Var']) + '_' +
                                '_'.join(values) + '.PNG')
        
        # save figure
        plt.savefig(figpath)
        
    def PlotAsymptoticResults(self):
        '''
        Generate plot of the asymptotic quantities attained by the algorithm
        after training
        '''

        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle(self.Param['outputModel'].split('_')[-1] + ' Asymptotic Results',fontsize=28)
        
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title(self.Param['Var'][0])
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title(self.Param['Var'][1])
        ax3 = fig.add_subplot(2,2,3)
        ax3.set_title(self.Param['Var'][2])
        ax4 = fig.add_subplot(2,2,4)
        ax4.set_title(self.Param['Var'][3])   
        
        
        if self.Param['Values'] == 'all':
            
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                       if os.path.isdir(os.path.join(self.savedpath,subdir))]
            
            values = self.Param['Values']
            
        else:
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
            values = [str(val) for val in self.Param['Values']]
            
        for folder in subdirs:
            
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
            
            # first plot
            ax1.plot(res_df[self.Param['Var'][0]].cumsum(), label=folder.split('_')[-1])
            #second plot
            ax2.plot(res_df[self.Param['Var'][1]].cumsum(), label=folder.split('_')[-1])
            #third plot
            ax3.plot(res_df[self.Param['Var'][2]].cumsum(), label=folder.split('_')[-1])
            #fourth plot
            ax4.plot(res_df[self.Param['Var'][3]].cumsum(), label=folder.split('_')[-1])
        
        
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        
        
        
        figpath = os.path.join(self.savedpath,'Qlearning_asympresults_'+ 
                                format_tousands(self.Param['N_train']) + '_' +
                                '_'.join(self.Param['Var']) + '_' +
                                '_'.join(values) + '.PNG')
        
        # save figure
        plt.savefig(figpath)


    def ComputeLearningSpeed(self,res_df,series_name):

        '''    
        Compute learning speed measure from a pnl series
        
        Parameters
        ----------
        series : pandas series or dataframe
            series of expected returns (pnl)
        
        Returns
        -------
        series which represent the learning speed obtained by the algorithm
        '''
        pnlmean = res_df[series_name].cumsum() / (res_df.index + 1)

        series = pnlmean.iloc[-1] - pnlmean
        
        return series
     