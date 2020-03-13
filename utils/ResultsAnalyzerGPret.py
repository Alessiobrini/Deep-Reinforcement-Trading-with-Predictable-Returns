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

import os, pdb
from utils.format_tousands import format_tousands


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
        
        '''
        Create proper directory path for accessing results. There is a flag for distinguish 
        experiments that are stored in single folder than those stored in the same
        folder (varying parameters). The path is then stored as attribute to the class
        '''
        
        if self.Param['varying_par']:
            savedpath = os.path.join(os.getcwd(),
                                     self.Param['outputDir'],
                                     self.Param['outputClass'],
                                     '_'.join([self.Param['outputModel'],self.Param['varying_par']]),
                                     format_tousands(self.Param['N_train']))
            
        else:
            savedpath = os.path.join(os.getcwd(),
                                     self.Param['outputDir'],
                                     self.Param['outputClass'],
                                     self.Param['outputModel'],
                                     format_tousands(self.Param['N_train']))
        
        self.savedpath = savedpath

    
    # ----------------------------------------------------------------------------
    # Public method       
    # ----------------------------------------------------------------------------


    def PlotLearningSpeed(self):
        
        '''
        Generate plot of the rate of learning of the algorithm
        after training. It can plot many example with varying parameters and no
        single experiment
        '''
        
        # instantiate figure
        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle('_'.join([self.Param['outputModel'],self.Param['varying_par']]) 
                               + ' Learning Speeds',fontsize=28)
        
        # add subplots to the figure with title equal to the variable to plot
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title(self.Param['SpeedVar'][0])
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title(self.Param['SpeedVar'][1])
        ax3 = fig.add_subplot(2,2,3)
        ax3.set_title(self.Param['SpeedVar'][2])
        ax4 = fig.add_subplot(2,2,4)
        ax4.set_title(self.Param['SpeedVar'][3])
        

        # select experiment to plot depending on the varing parameters provided
        # 'all' will take every subfolder
        # if no varying parameter is selected, there will be no subdirs
        # the structure of the folder will be 
        # outputDir/outputClass/outputModel/Numberiteration/subfolders
        if self.Param['Values'] == 'all':
            
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                       if os.path.isdir(os.path.join(self.savedpath,subdir))]
            
            # store values as string for filename
            values = self.Param['Values']
            
        else:
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
            values = [str(val) for val in self.Param['Values']]

        # iterate over subdirectory
        for folder in subdirs:
            
            # read parquet file with results
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
            
            # plot speeds for the 4 subplots 
            # first plot
            speed1 = self.ComputeLearningSpeed(res_df,self.Param['SpeedVar'][0])
            ax1.plot(speed1.iloc[self.Param['start']:], label=folder.split('_')[-1])
            #second plot
            speed2 = self.ComputeLearningSpeed(res_df,self.Param['SpeedVar'][1])
            ax2.plot(speed2.iloc[self.Param['start']:], label=folder.split('_')[-1])
            #third plot
            speed3 = self.ComputeLearningSpeed(res_df,self.Param['SpeedVar'][2])
            ax3.plot(speed3.iloc[self.Param['start']:], label=folder.split('_')[-1])
            #fourth plot
            speed4 = self.ComputeLearningSpeed(res_df,self.Param['SpeedVar'][3])
            ax4.plot(speed4.iloc[self.Param['start']:], label=folder.split('_')[-1])
            
            if self.Param['plot_speed_post'] == 1:
                
                # read parquet file with post results
                res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_Test' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
                
                # plot post speeds for the 4 subplots
                # first plot
                speed1_post = self.ComputeLearningSpeed(res_df_post,self.Param['SpeedVar'][0])
                ax1.plot(speed1_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                #second plot
                speed2_post = self.ComputeLearningSpeed(res_df_post,self.Param['SpeedVar'][1])
                ax2.plot(speed2_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                #third plot
                speed3_post = self.ComputeLearningSpeed(res_df_post,self.Param['SpeedVar'][2])
                ax3.plot(speed3_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                #fourth plot
                speed4_post = self.ComputeLearningSpeed(res_df_post,self.Param['SpeedVar'][3])
                ax4.plot(speed4_post.iloc[self.Param['start']:], label=folder.split('_')[-1] + '_post')
                
        
        # insert legend in the subplots
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        
        
        # generate figure filename
        figpath = os.path.join(self.savedpath,'Qlearning_speeds_'+ 
                                format_tousands(self.Param['N_train']) + '_' +
                                '_'.join(self.Param['SpeedVar']) + '_' +
                                '_'.join(values) + '.PNG')
        
        # save figure
        plt.savefig(figpath)
        
    def PlotAsymptoticResults(self):
        '''
        Generate plot of the asymptotic quantities attained by the algorithm
        after training. It can plot many example with varying parameters and
        no single experiment, since the asymptotic results can be already seen from
        the runtime output
        '''
        # instantiate figure
        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        plt.suptitle(self.Param['outputModel'].split('_')[-1] + ' Asymptotic Results',fontsize=28)
        
        
        # add subplots to the figure with title equal to the variable to plot
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title(self.Param['AsymVar'][0])
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title(self.Param['AsymVar'][1])
        ax3 = fig.add_subplot(2,2,3)
        ax3.set_title(self.Param['AsymVar'][2])
        ax4 = fig.add_subplot(2,2,4)
        ax4.set_title(self.Param['AsymVar'][3])   
        
        # select experiment to plot depending on the varing parameters provided
        # 'all' will take every subfolder
        # if no varying parameter is selected, there will be no subdirs
        # the structure of the folder will be 
        # outputDir/outputClass/outputModel/Numberiteration/subfolders
        if self.Param['Values'] == 'all':
            
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                       if os.path.isdir(os.path.join(self.savedpath,subdir))]
            
            # store values as string for filename
            values = self.Param['Values']
            
        else:
            subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
            values = [str(val) for val in self.Param['Values']]
        
        
        # iterate over subdirectory
        for folder in subdirs:
            
            # read parquet file with results
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
            
            # plot asym results for the 4 subplots 
            # first plot
            ax1.plot(res_df[self.Param['AsymVar'][0]].cumsum(), label=folder.split('_')[-1])
            #second plot
            ax2.plot(res_df[self.Param['AsymVar'][1]].cumsum(), label=folder.split('_')[-1])
            #third plot
            ax3.plot(res_df[self.Param['AsymVar'][2]].cumsum(), label=folder.split('_')[-1])
            #fourth plot
            ax4.plot(res_df[self.Param['AsymVar'][3]].cumsum(), label=folder.split('_')[-1])
        
        # insert legend in the subplots
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        
        # generate figure filename
        figpath = os.path.join(self.savedpath,'Qlearning_asympresults_'+ 
                                format_tousands(self.Param['N_train']) + '_' +
                                '_'.join(self.Param['AsymVar']) + '_' +
                                '_'.join(values) + '.PNG')
        
        # save figure
        plt.savefig(figpath)
        
    def PlotActions(self):
        
        '''
        Generate plot of action and portfolio of the algorithm during and
        after training. Better to be used with only one experiment at a time.
        Therefore even if the model is in a folder with a varying parameter,
        choose only one value in the config, otherwise the plot would be unreadable.
        '''
        
        # instantiate figure
        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        
        # set the title for the plot depeding on the flag varying param
        if self.Param['varying_par']:
            plt.suptitle('_'.join([self.Param['outputModel'],self.Param['varying_par']]) 
                                   + ' actions and portfolio',fontsize=28)            
        else:
            plt.suptitle(self.Param['outputModel']+ ' actions and portfolio',fontsize=28)
                       
        # add subplots to the figure with title equal to the variable to plot
        ax1 = fig.add_subplot(4,1,1)
        ax1.set_title(self.Param['ActVar'][0],x=-0.1,y=0.5)
        ax2 = fig.add_subplot(4,1,2)
        ax2.set_title(self.Param['ActVar'][1],x=-0.1,y=0.5)
        ax3 = fig.add_subplot(4,1,3)
        ax3.set_title(self.Param['ActVar'][2],x=-0.1,y=0.5)
        ax4 = fig.add_subplot(4,1,4)
        ax4.set_title(self.Param['ActVar'][3],x=-0.1,y=0.5)
        
        
        # select experiment to plot depending on the varing parameters provided
        # if no varying parameter is selected, there will be no subdirs
        # the structure of the folder will be 
        # outputDir/outputClass/outputModel/Numberiteration/subfolders
        # in the case of no subdirs, the flag for the single experiment is activated
        subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
        values = [str(val) for val in self.Param['Values']]
        
        
        # if there is no subfolder it mean there is only one experiment ran in the savedpath folder
        if subdirs:
            
            for folder in subdirs:
                res_df = pd.read_parquet(os.path.join(self.savedpath,
                                                      folder,'Results_' +
                                                      format_tousands(self.Param['N_train']) 
                                                      + '.parquet.gzip'))
                
                # first plot
                ax1.plot(res_df[self.Param['ActVar'][0]].iloc[self.Param['start']:self.Param['end']],
                         label=folder.split('_')[-1],
                         color = 'blue')
                #second plot
                ax2.plot(res_df[self.Param['ActVar'][1]].iloc[self.Param['start']:self.Param['end']],
                         label=folder.split('_')[-1],
                         color = 'blue')
                #third plot
                ax3.plot(res_df[self.Param['ActVar'][2]].iloc[self.Param['start']:self.Param['end']],
                         label=folder.split('_')[-1],
                         color = 'blue')
                #fourth plot
                ax4.plot(res_df[self.Param['ActVar'][3]].iloc[self.Param['start']:self.Param['end']],
                         label=folder.split('_')[-1],
                         color = 'blue')
                
                # plot post results
                if self.Param['plot_action_post'] == 1:
                    
                    res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                                  folder,'Results_Test' +
                                                  format_tousands(self.Param['N_train']) 
                                                  + '.parquet.gzip'))
                    
                    # first plot
                    ax1.plot(res_df_post[self.Param['ActVar'][0]].iloc[self.Param['start']:self.Param['end']],
                             label=folder.split('_')[-1] + '_post',
                             color = 'orange',
                             alpha = 0.7)
                    #second plot
                    ax2.plot(res_df_post[self.Param['ActVar'][1]].iloc[self.Param['start']:self.Param['end']],
                             label=folder.split('_')[-1] + '_post',
                             color = 'orange',
                             alpha = 0.7)
                    #third plot
                    ax3.plot(res_df_post[self.Param['ActVar'][2]].iloc[self.Param['start']:self.Param['end']],
                             label=folder.split('_')[-1] + '_post',
                             color = 'orange',
                             alpha = 0.7)
                    #fourth plot
                    ax4.plot(res_df_post[self.Param['ActVar'][3]].iloc[self.Param['start']:self.Param['end']],
                             label=folder.split('_')[-1] + '_post',
                             color = 'orange',
                             alpha = 0.7)
                    
                figpath = os.path.join(self.savedpath,'Qlearning_actions_'+ 
                                        format_tousands(self.Param['N_train']) + '_' +
                                        '_'.join(self.Param['ActVar']) + '_' +
                                        '_'.join(values) + '.PNG')
        
        # case in which there is only an experiment in the folder
        else:
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                                  'Results_' +
                                                  format_tousands(self.Param['N_train']) 
                                                  + '.parquet.gzip'))                
            # first plot
            ax1.plot(res_df[self.Param['ActVar'][0]].iloc[self.Param['start']:self.Param['end']],
                     label=self.Param['ActVar'][0],
                     color = 'blue')
            #second plot
            ax2.plot(res_df[self.Param['ActVar'][1]].iloc[self.Param['start']:self.Param['end']],
                     label=self.Param['ActVar'][1],
                     color = 'blue')
            #third plot
            ax3.plot(res_df[self.Param['ActVar'][2]].iloc[self.Param['start']:self.Param['end']],
                     label=self.Param['ActVar'][2],
                     color = 'blue')
            #fourth plot
            ax4.plot(res_df[self.Param['ActVar'][3]].iloc[self.Param['start']:self.Param['end']],
                     label=self.Param['ActVar'][3],
                     color = 'blue')
            
            # plot post results
            if self.Param['plot_action_post'] == 1:
                
                res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                              'Results_Test' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
                
                # first plot
                ax1.plot(res_df_post[self.Param['ActVar'][0]].iloc[self.Param['start']:self.Param['end']],
                         label=self.Param['ActVar'][0] + '_post',
                         color = 'orange',
                         alpha = 0.7)
                #second plot
                ax2.plot(res_df_post[self.Param['ActVar'][1]].iloc[self.Param['start']:self.Param['end']],
                         label=self.Param['ActVar'][1] + '_post',
                         color = 'orange',
                         alpha = 0.7)
                #third plot
                ax3.plot(res_df_post[self.Param['ActVar'][2]].iloc[self.Param['start']:self.Param['end']],
                         label=self.Param['ActVar'][2] + '_post',
                         color = 'orange',
                         alpha = 0.7)
                #fourth plot
                ax4.plot(res_df_post[self.Param['ActVar'][3]].iloc[self.Param['start']:self.Param['end']],
                         label=self.Param['ActVar'][3] + '_post',
                         color = 'orange',
                         alpha = 0.7)
            
            figpath = os.path.join(self.savedpath,'Qlearning_actions_'+ 
                                    format_tousands(self.Param['N_train']) + '_' +
                                    self.Param['outputModel'] + 
                                    '_'.join(self.Param['ActVar']) + '_' + '.PNG')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
           
        # save figure
        plt.savefig(figpath)



    def PlotScatterCorrs(self):
        
        '''
        Generate scatter plots of action and portfolio of the algorithm during and
        after training. Better to be used with only one experiment at a time.
        Therefore even if the model is in a folder with a varying parameter,
        choose only one value in the config, otherwise the plot would be unreadable.
        '''

        fig = plt.figure(figsize=(34,13))
        fig.tight_layout()
        
        if self.Param['varying_par']:
            plt.suptitle('_'.join([self.Param['outputModel'],self.Param['varying_par']]) 
                                   + ' Scatter for Correlations',fontsize=28)            
        else:
            plt.suptitle(self.Param['outputModel']+ ' actions and portfolio',fontsize=28)
                       
            
        ax1 = fig.add_subplot(1,2,1)   
        ax2 = fig.add_subplot(1,2,2)

        # same reasoning of the previous method, except for the fact that it is
        # better to plot one experiment at a time. So even if with varying parameter
        # choose only one value per run
        subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
        values = [str(val) for val in self.Param['Values']]
        
        
        # if there is no subfolder it mean there is only one experiment ran in the savedpath folder
        if subdirs:
            
            for folder in subdirs:
                res_df = pd.read_parquet(os.path.join(self.savedpath,
                                                      folder,'Results_' +
                                                      format_tousands(self.Param['N_train']) 
                                                      + '.parquet.gzip'))
                
                # first scatter plot
                ax1.scatter(res_df[self.Param['CorrVar'][0][0]].iloc[self.Param['start']:self.Param['end']],
                            res_df[self.Param['CorrVar'][0][1]].iloc[self.Param['start']:self.Param['end']],
                            label=folder.split('_')[-1],
                            color = 'blue')
                
                ax1.set_xlabel(self.Param['CorrVar'][0][0])
                ax1.set_ylabel(self.Param['CorrVar'][0][1])
                
                #second scatter plot
                ax2.scatter(res_df[self.Param['CorrVar'][0][0]].iloc[self.Param['start']:self.Param['end']],
                            res_df[self.Param['CorrVar'][0][1]].iloc[self.Param['start']:self.Param['end']],
                            label=folder.split('_')[-1],
                            color = 'blue')
                
                ax2.set_xlabel(self.Param['CorrVar'][1][0])
                ax2.set_ylabel(self.Param['CorrVar'][1][1])

                
                # scatter plot post results
                if self.Param['plot_action_post'] == 1:
                    
                    res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                                  folder,'Results_Test' +
                                                  format_tousands(self.Param['N_train']) 
                                                  + '.parquet.gzip'))
                    
                    # first plot
                    ax1.scatter(res_df_post[self.Param['CorrVar'][0][0]].iloc[self.Param['start']:self.Param['end']],
                                res_df_post[self.Param['CorrVar'][0][1]].iloc[self.Param['start']:self.Param['end']],
                                label=folder.split('_')[-1] + '_post',
                                color = 'orange',
                                alpha = 0.7)
                    #second plot
                    ax2.scatter(res_df_post[self.Param['CorrVar'][1][0]].iloc[self.Param['start']:self.Param['end']],
                                res_df_post[self.Param['CorrVar'][1][1]].iloc[self.Param['start']:self.Param['end']],
                                label=folder.split('_')[-1] + '_post',
                                color = 'orange',
                                alpha = 0.7)

                figpath = os.path.join(self.savedpath,'Qlearning_scattercorr_'+ 
                                        format_tousands(self.Param['N_train']) + '_' +
                                        '_'.join(self.Param['ActVar']) + '_' +
                                        '_'.join(values) + '.PNG')
        
        # case in which there is only an experiment in the folder
        else:
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                                  'Results_' +
                                                  format_tousands(self.Param['N_train']) 
                                                  + '.parquet.gzip'))                
            # first plot
            ax1.scatter(res_df[self.Param['CorrVar'][0][0]].iloc[self.Param['start']:self.Param['end']],
                        res_df[self.Param['CorrVar'][0][1]].iloc[self.Param['start']:self.Param['end']],
                        label=self.Param['CorrVar'][0],
                        color = 'blue')
            
            ax1.set_xlabel(self.Param['CorrVar'][0][0])
            ax1.set_ylabel(self.Param['CorrVar'][0][1])
            
            #second plot
            ax2.scatter(res_df[self.Param['CorrVar'][1][0]].iloc[self.Param['start']:self.Param['end']],
                        res_df[self.Param['CorrVar'][1][1]].iloc[self.Param['start']:self.Param['end']],
                        label=self.Param['CorrVar'][1],
                        color = 'blue')
            
            ax2.set_xlabel(self.Param['CorrVar'][1][0])
            ax2.set_ylabel(self.Param['CorrVar'][1][1])

            # plot post results
            if self.Param['plot_action_post'] == 1:
                
                res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_Test' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
                
                # first plot
                ax1.scatter(res_df_post[self.Param['CorrVar'][0][0]].iloc[self.Param['start']:self.Param['end']],
                            res_df_post[self.Param['CorrVar'][0][1]].iloc[self.Param['start']:self.Param['end']],
                            label=self.Param['ActVar'][0] + '_post',
                            color = 'orange',
                            alpha = 0.7)
                
                ax1.set_xlabel(self.Param['CorrVar'][0][0])
                ax1.set_ylabel(self.Param['CorrVar'][0][1])
                
                #second plot
                ax2.scatter(res_df_post[self.Param['CorrVar'][1][0]].iloc[self.Param['start']:self.Param['end']],
                            res_df_post[self.Param['CorrVar'][1][1]].iloc[self.Param['start']:self.Param['end']],
                            label=self.Param['ActVar'][1] + '_post',
                            color = 'orange',
                            alpha = 0.7)
               
                ax2.set_xlabel(self.Param['CorrVar'][1][0])
                ax2.set_ylabel(self.Param['CorrVar'][1][1])
            
            figpath = os.path.join(self.savedpath,'Qlearning_actions_'+ 
                                    format_tousands(self.Param['N_train']) + '_' +
                                    self.Param['outputModel'] + 
                                    '_'.join(self.Param['ActVar']) + '_' + '.PNG')

        ax1.legend()
        ax2.legend()
        
        # save figure
        plt.savefig(figpath)
        
    def ComputeCorrs(self):
        
        '''
        Generate plot of action and portfolio of the algorithm during and
        after training. Better to be used with only one experiment at a time.
        Therefore even if the model is in a folder with a varying parameter,
        choose only one value in the config, otherwise the plot would be unreadable.
        '''

        start = self.Param['start']
        end = self.Param['end']
        steps = self.Param['step']
        
        corr = pd.DataFrame(columns = ['_'.join(tup) for tup in self.Param['CorrVar']],
                            index = np.arange(start,
                                              end + 1,
                                              steps)).drop(0)
        
        
        subdirs = [subdir for subdir in os.listdir(self.savedpath) 
                   if (os.path.isdir(os.path.join(self.savedpath,subdir)) and 
                   float(subdir.split('_')[-1]) in self.Param['Values'])]
            
        values = [str(val) for val in self.Param['Values']]
        
        
        # if there is no subfolder it mean there is only one experiment ran in the savedpath folder
        if subdirs:
            
            for folder in subdirs:
                res_df = pd.read_parquet(os.path.join(self.savedpath,
                                                      folder,'Results_' +
                                                      format_tousands(self.Param['N_train']) 
                                                      + '.parquet.gzip'))

                for step in corr.index:
                    for tup in self.Param['CorrVar']:

                        # # expanding window
                        # corr.at[step,'_'.join(tup)] = np.corrcoef(np.sign(res_df[tup[0]].iloc[start:step]),
                        #                                           np.sign(res_df[tup[1]].iloc[start:step]))[0][1] 
                        corr.at[step,'_'.join(tup)] = np.corrcoef(res_df[tup[0]].iloc[start:step],
                                                                  res_df[tup[1]].iloc[start:step])[0][1] 
                        # print(tup[0], ' mean: ', res_df[tup[0]].iloc[start:step].mean())
                        # print(tup[1], ' mean: ', res_df[tup[1]].iloc[start:step].mean())
                # plot post results
                if self.Param['corr_action_post'] == 1:
                    
                    res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                                  folder,'Results_Test' +
                                                  format_tousands(self.Param['N_train']) 
                                                  + '.parquet.gzip'))
                    
                    #pdb.set_trace()
                    newcols = ['_'.join(tup) + '_post' for tup in self.Param['CorrVar']]
                    for i in range(len(newcols)):
                        corr[newcols[i]] = 0
                    #pdb.set_trace()
                    for step in corr.index:
                        for tup in newcols:
                            # expanding window
                            corr.at[step, tup] = np.corrcoef(res_df_post[tup.split('_')[0]].iloc[start:step],
                                                             res_df_post[tup.split('_')[1]].iloc[start:step])[0][1]
            
            if self.Param['print_corr']:
                print(corr.to_latex())
                
            # save file
            # corr.to_parquet(os.path.join(self.savedpath,
            #                   'Corr_' + format_tousands(self.Param['N_train']) + '_' +
            #                   self.Param['outputModel'] + '_' + '_'.join(values)+ '_'.join(corr.columns) + '_'
            #                   + '_'.join(['start',str(self.Param['start']),
            #                               'end',str(self.Param['end']),
            #                               'step',str(self.Param['step'])])
            #                               + '.parquet.gzip'),compression='gzip')
                    
        # case in which there is only an experiment in the folder
        else:
            res_df = pd.read_parquet(os.path.join(self.savedpath,
                                                  'Results_' +
                                                  format_tousands(self.Param['N_train']) 
                                                  + '.parquet.gzip'))                

            for step in corr.index:
                for tup in self.Param['CorrVar']:
                    # expanding window
                    corr.at[step,'_'.join(tup)] = np.corrcoef(res_df[tup[0]].iloc[start:step],
                                                              res_df[tup[1]].iloc[start:step])[0][1]
            # plot post results
            if self.Param['corr_action_post'] == 1:
                
                res_df_post = pd.read_parquet(os.path.join(self.savedpath,
                                              folder,'Results_Test' +
                                              format_tousands(self.Param['N_train']) 
                                              + '.parquet.gzip'))
                
                newcols = ['_'.join(tup) + '_post' for tup in self.Param['CorrVar']]
                for i in range(len(newcols)):
                    corr[newcols[i]] = 0
                     
                for step in corr.index:
                    for tup in newcols:
                        # expanding window
                        corr.at[step,tup] = np.corrcoef(res_df_post[tup.split('_')[0]].iloc[start:step],
                                                        res_df_post[tup.split('_')[1]].iloc[start:step])[0][1]
            
            if self.Param['print_corr']:
                print(corr)
                
            # save file
            # corr.to_parquet(os.path.join(self.savedpath,
            #                   'Corr_' + format_tousands(self.Param['N_train']) + '_' +
            #                   self.Param['outputModel'] + '_' + '_'.join(values) + '_'.join(corr.columns) + '_'
            #                   + '_'.join(['start',str(self.Param['start']),
            #                               'end',str(self.Param['end']),
            #                               'step',str(self.Param['step'])])
            #                               + '.parquet.gzip'),compression='gzip')
            
            
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

        series = (pnlmean.iloc[-1] - pnlmean) / pnlmean.iloc[-1]
        
        return series
     