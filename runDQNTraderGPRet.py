# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:45:22 2019

@author: aless
"""
import os

if any('SPYDER' in name for name in os.environ):
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    

# 0. importing section initialize logger.--------------------------------------
import logging, os, pdb, sys
from utils.readYaml import readConfigYaml, saveConfigYaml 
from utils.generateLogger import generate_logger
from utils.SavePath import GeneratePathFolder
from utils.SimulateData import ReturnSampler
from utils.MarketEnv import MarketEnv, ActionSpace
from utils.DeepLearning import DQN
from utils.DQNOutputs import PlotLearningResults
from tqdm import tqdm
import datetime
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import seaborn 
seaborn.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['savefig.dpi'] = 90
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

# 0. Generate Logger-------------------------------------------------------------
logger = generate_logger()

# 1. Read config ---------------------------------------------------------------- 
# maybe substitute with argparse
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramDQNGPRet.yaml'))
logging.info('Successfully read config file with parameters...')

#DQN implementation
epsilon = Param['epsilon']
eps_decay = Param['eps_decay']
min_eps = Param['min_eps']
alpha = Param['alpha']
gamma = Param['gamma']
kappa = Param['kappa']
activation = Param['activation']
kernel_initializer = Param['kernel_initializer']
batch_norm = Param['batch_norm']
hidden_units = Param['hidden_units']
batch_size = Param['batch_size']
max_experiences = Param['max_experiences']
min_experiences = Param['min_experiences']
copy_step = Param['copy_step']
learning_rate = Param['learning_rate']
LotSize = Param['LotSize']
K = Param['K']
HalfLife = Param['HalfLife']
# Data Simulation
f0 = Param['f0']
f_param = Param['f_param']
sigma = Param['sigma']
sigmaf = Param['sigmaf']
CostMultiplier = Param['CostMultiplier']
discount_rate = Param['discount_rate']
Startholding = Param['Startholding']
seed = Param['Seed']
# Experiment and storage
N_train = Param['N_train']
plot_inputs = Param['plot_inputs']
plot_insample = Param['plot_insample']
executeRL = Param['executeRL']
execute_GP = Param['execute_GP']
save_results = Param['save_results']
plot_steps = Param['plot_steps']
outputDir = Param['outputDir']
outputClass = Param['outputClass']
outputModel = Param['outputModel']
varying_par = Param['varying_par']

# 2. Generate path for model and tensorboard output, store config ---------------
savedpath = GeneratePathFolder(outputDir, outputClass, outputModel, varying_par, N_train, Param)
saveConfigYaml(Param,savedpath)
#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
log_dir = os.path.join(savedpath,'tb')
summary_writer = tf.summary.create_file_writer(log_dir)
logging.info('Successfully generated path and stored config...')


# 2. Generate data --------------------------------------------------------------
returns, factors, f_speed = ReturnSampler(N_train, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, seed)
logging.info('Successfully simulated data...YOU ARE CURRENTLY USING A SEED TO SIMULATE RETURNS. LEAVE IT IF YOU HAVE FOUND A PROPER NN SETTING')
# 3. Instantiate Env --------------------------------------------------------------
action_space = ActionSpace(K,LotSize)
env = MarketEnv(K, LotSize,  HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                N_train, plot_insample, discount_rate, f_param, f_speed, returns, factors)
logging.info('Successfully initialized the market environment...')

# 4. Train DQN algorithm ----------------------------------------------------------
CurrState = env.reset()
if execute_GP:
    CurrOptState = env.opt_reset()
    OptRate = env.opt_trading_rate()
iters = 0
num_states = len(CurrState)

TrainNet = DQN(num_states, hidden_units, gamma, max_experiences, 
               min_experiences, batch_size, learning_rate, action_space, batch_norm, summary_writer, activation, kernel_initializer,plot_steps)
TargetNet = DQN(num_states, hidden_units, gamma, max_experiences, 
                min_experiences, batch_size, learning_rate, action_space, batch_norm, summary_writer, activation, kernel_initializer,plot_steps)
BaselineNet = DQN(num_states, hidden_units, gamma, max_experiences, 
               min_experiences, batch_size, learning_rate, action_space, batch_norm, summary_writer, activation, kernel_initializer,plot_steps)
logging.info('Successfully initialized the Deep Q Networks...YOU ARE CURRENTLY USING A SEED TO INITIALIZE WEIGHTS. LEAVE IT IF YOU HAVE FOUND A PROPER NN SETTING')

for i in tqdm(iterable=range(N_train + 1), desc='Training DQNetwork'):
    
    if executeRL:
        epsilon = max(min_eps, epsilon - eps_decay) # linear decay
        shares_traded = TrainNet.eps_greedy_action(CurrState, epsilon)
        NextState, Result = env.step(CurrState, shares_traded, i)
        env.store_results(Result, i)


        exp = {'s': CurrState, 'a': shares_traded, 'r': Result['Reward'], 's2': NextState}
        TrainNet.add_experience(exp)
        if i == min_experiences:
            TrainNet.add_test_experience(exp, i)
        TrainNet.train(TargetNet,BaselineNet, i)
        TrainNet.test(BaselineNet, i)
            
        CurrState = NextState
        iters += 1
        if iters % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        
        if plot_insample:
            if (i % (N_train/5) == 0) & (i != 0):
                PlotLearningResults(res_df=env.res_df.loc[:i], 
                                    title='DQN_iteration_' + str(i),
                                    iteration=i,
                                    executeRL=executeRL,
                                    savedpath=savedpath,
                                    N_train=N_train)
    
    
    if execute_GP:
        NextOptState, OptResult = env.opt_step(CurrOptState, OptRate, i)
        env.store_results(OptResult, i)
        
        # with summary_writer.as_default():
        #     for key in OptResult.keys():
        #         tf.summary.scalar(key, OptResult[key], step=i)
        #         summary_writer.flush()
        
        CurrOptState = NextOptState
logging.info('Successfully trained the Deep Q Network...')
# 5. Plot and save outputs ----------------------------------------------------------     
if save_results:
    env.save_outputs(savedpath)

if execute_GP:
    PlotLearningResults(res_df=env.res_df, 
                        title='DQN_benchmark',
                        iteration = N_train,
                        executeRL=executeRL,
                        savedpath=savedpath,
                        N_train=N_train,
                        plot_GP=execute_GP)
logging.info('Successfully plotted and stored results...')
