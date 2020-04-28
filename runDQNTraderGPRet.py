# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:45:22 2019

@author: aless
"""
# delete any variables created in previous run if you are using this script on Spyder
import os
if any('SPYDER' in name for name in os.environ):
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    

# 0. importing section initialize logger.--------------------------------------
import logging, os, pdb
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

# 0. Generate Logger-------------------------------------------------------------
logger = generate_logger()

# 1. Read config ---------------------------------------------------------------- 
# maybe substitute with argparse
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramDQNGPRet.yaml'))
logging.info('Successfully read config file with parameters...')

def RunDQN(Param):
    
    #DQN implementation
    epsilon = Param['epsilon']
    eps_decay = Param['eps_decay']
    min_eps = Param['min_eps']
    gamma = Param['gamma']
    kappa = Param['kappa']
    DQN_type = Param['DQN_type']
    selected_loss = Param['selected_loss']
    activation = Param['activation']
    kernel_initializer = Param['kernel_initializer']
    batch_norm_input = Param['batch_norm_input']
    batch_norm_hidden = Param['batch_norm_hidden']
    mom_batch_norm  = Param['mom_batch_norm']
    trainable_batch_norm = Param['trainable_batch_norm']
    clipgrad = Param['clipgrad']
    clipnorm = Param['clipnorm']
    clipvalue = Param['clipvalue']
    optimizer_name = Param['optimizer_name']
    optimizer_decay = Param['optimizer_decay']
    beta_1 = Param['beta_1']
    beta_2 = Param['beta_2']
    eps_opt = Param['eps_opt']
    hidden_units = Param['hidden_units']
    batch_size = Param['batch_size']
    max_experiences = Param['max_experiences']
    min_experiences = Param['min_experiences']
    copy_step = Param['copy_step']
    learning_rate = Param['learning_rate']
    lr_schedule = Param['lr_schedule']
    exp_decay_steps = Param['exp_decay_steps']
    exp_decay_rate = Param['exp_decay_rate']
    KL = Param['KL']
    HalfLife = Param['HalfLife']
    # Data Simulation
    f0 = Param['f0']
    f_param = Param['f_param']
    sigma = Param['sigma']
    sigmaf = Param['sigmaf']
    CostMultiplier = Param['CostMultiplier']
    discount_rate = Param['discount_rate']
    Startholding = Param['Startholding']
    seed = Param['seed']
    # Experiment and storage
    N_train = Param['N_train']
    plot_inputs = Param['plot_inputs']
    plot_insample = Param['plot_insample']
    executeRL = Param['executeRL']
    execute_GP = Param['execute_GP']
    save_results = Param['save_results']
    plot_hist = Param['plot_hist']
    plot_steps_hist = Param['plot_steps_hist']
    plot_steps = Param['plot_steps']
    save_model = Param['save_model']
    runtype = Param['runtype']
    outputDir = Param['outputDir']
    outputClass = Param['outputClass']
    outputModel = Param['outputModel']
    varying_pars = Param['varying_pars']
    # varying_par = Param['varying_par']
    # varying_par2 = Param['varying_par2']
    
    # 2. Generate path for model and tensorboard output, store config ---------------
    savedpath = GeneratePathFolder(outputDir, outputClass, outputModel, varying_pars, N_train, Param)
    saveConfigYaml(Param,savedpath)
    log_dir = os.path.join(savedpath, 'tb')
        
    summary_writer = tf.summary.create_file_writer(log_dir)
    logging.info('Successfully generated path and stored config...')
    
    # 2. Generate data --------------------------------------------------------------
    returns, factors, f_speed = ReturnSampler(N_train, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, seed)
    logging.info('Successfully simulated data...YOU ARE CURRENTLY USING A SEED TO SIMULATE RETURNS. LEAVE IT IF YOU HAVE FOUND A PROPER NN SETTING')
    # 3. Instantiate Env --------------------------------------------------------------
    action_space = ActionSpace(KL)
    env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                    N_train, plot_insample, discount_rate, f_param, f_speed, returns, factors)
    logging.info('Successfully initialized the market environment...')
    
    # 4. Train DQN algorithm ----------------------------------------------------------
    # instantiate the initial state (return, holding)
    CurrState = env.reset()
    # instantiate the initial state for the benchmark, if required
    if execute_GP:
        CurrOptState = env.opt_reset()
        OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
    # iteration count to decide when copying weights for the Target Network
    iters = 0
    num_states = len(CurrState)
    
    TrainNet = DQN(seed, num_states, hidden_units, gamma, max_experiences, 
                   min_experiences, batch_size, learning_rate, lr_schedule, exp_decay_steps, exp_decay_rate, 
                   action_space, batch_norm_input, batch_norm_hidden, 
                   summary_writer, activation, kernel_initializer, plot_hist, plot_steps_hist, plot_steps,selected_loss, 
                   mom_batch_norm, trainable_batch_norm, DQN_type, clipgrad, clipnorm, 
                   clipvalue, optimizer_name, optimizer_decay, beta_1, beta_2, eps_opt, modelname='TrainNet')
    
    TargetNet = DQN(seed, num_states, hidden_units, gamma, max_experiences, 
                    min_experiences, batch_size, learning_rate, lr_schedule, exp_decay_steps, exp_decay_rate,
                    action_space, batch_norm_input, batch_norm_hidden, 
                    summary_writer, activation, kernel_initializer, plot_hist, plot_steps_hist, plot_steps,selected_loss, 
                    mom_batch_norm, trainable_batch_norm, DQN_type, clipgrad, clipnorm, 
                    clipvalue, optimizer_name, optimizer_decay, beta_1, beta_2, eps_opt, modelname='TargetNet')
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
                TrainNet.add_test_experience()
                TrainNet.compute_test_target(TargetNet)
    
            
            TrainNet.train(TargetNet, i)
            TrainNet.test(TargetNet, i)
                
            CurrState = NextState
            iters += 1
            if (iters % copy_step == 0) and (i > TrainNet.min_experiences):
                TargetNet.copy_weights(TrainNet)
                TrainNet.compute_test_target(TargetNet)
                if execute_GP:
                    TrainNet.compute_portfolio_distance(env, OptRate, DiscFactorLoads, i)
            
            if plot_insample:
                if (i % (N_train/5) == 0) & (i != 0):
                    PlotLearningResults(res_df=env.res_df.loc[:i], 
                                        title='DQN_iteration_' + str(i),
                                        iteration=i,
                                        executeRL=executeRL,
                                        savedpath=savedpath,
                                        N_train=N_train)
        
        
        if execute_GP:
            NextOptState, OptResult = env.opt_step(CurrOptState, OptRate, DiscFactorLoads, i)
            env.store_results(OptResult, i) 
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
    
    if save_model:
        TrainNet.model.save_weights(os.path.join(savedpath,'DQN_weights'), save_format='tf')
        logging.info('Successfully saved DQN weights...')


if __name__ == "__main__":
    RunDQN(Param)