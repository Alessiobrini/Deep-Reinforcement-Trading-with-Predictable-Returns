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
import logging, os, pdb, sys
from utils.readYaml import readConfigYaml, saveConfigYaml 
from utils.generateLogger import generate_logger
from utils.SavePath import GeneratePathFolder
from utils.SimulateData import ReturnSampler
from utils.TradersMarketEnv import MarketEnv, ActionSpace,ReturnSpace,HoldingSpace,CreateQTable
from utils.DeepLearning import DQN
from utils.PreTraining import PreTraining
# from utils.LaunchIpynbs import runNBs
from tqdm import tqdm
import tensorflow as tf
import numpy as np

# 0. Generate Logger-------------------------------------------------------------
logger = generate_logger()

# 1. Read config ---------------------------------------------------------------- 
# maybe substitute with argparse
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramTradersTest.yaml'))
logging.info('Successfully read config file with parameters...')

def RunTraders(Param):
    
    #DQN implementation
    epsilon = Param['epsilon']
    steps_to_min_eps = Param['steps_to_min_eps']
    min_eps = Param['min_eps']
    gamma = Param['gamma']
    kappa = Param['kappa']
    std_rwds = Param['std_rwds']
    DQN_type = Param['DQN_type']
    use_PER = Param['use_PER']
    PER_e = Param['PER_e']
    PER_a = Param['PER_a']
    PER_b = Param['PER_b']
    PER_b_anneal = Param['PER_b_anneal']
    final_PER_b = Param['final_PER_b']
    PER_b_steps = Param['PER_b_steps']
    PER_a_anneal = Param['PER_a_anneal']
    final_PER_a = Param['final_PER_a']
    PER_a_steps = Param['PER_a_steps']
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
    clipglob_steps = Param['clipglob_steps']
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
    update_target = Param['update_target']
    tau = Param['tau']
    learning_rate = Param['learning_rate']
    lr_schedule = Param['lr_schedule']
    exp_decay_steps = Param['exp_decay_steps']
    final_lr = Param['final_lr']
    KLM = Param['KLM']
    RT = Param['RT']
    tablr = Param['tablr']
    # Data Simulation
    HalfLife = Param['HalfLife']
    f0 = Param['f0']
    f_param = Param['f_param']
    sigma = Param['sigma']
    sigmaf = Param['sigmaf']
    CostMultiplier = Param['CostMultiplier']
    discount_rate = Param['discount_rate']
    Startholding = Param['Startholding']
    # Experiment and storage
    start_train = Param['start_train']
    seed = Param['seed']
    N_train = Param['N_train']
    out_of_sample_test = Param['out_of_sample_test']
    N_test = Param['N_test']
    plot_inputs = Param['plot_inputs']
    executeDRL = Param['executeDRL']
    executeRL = Param['executeRL']
    executeGP = Param['executeGP']
    save_results = Param['save_results']
    save_table = Param['save_table']
    plot_hist = Param['plot_hist']
    plot_steps_hist = Param['plot_steps_hist']
    plot_steps = Param['plot_steps']
    pdist_steps = Param['pdist_steps']
    save_model = Param['save_model']
    save_ckpt_model = Param['save_ckpt_model']
    saveresultsNBs = Param['saveresultsNBs']
    use_GPU = Param['use_GPU']
    outputDir = Param['outputDir']
    outputClass = Param['outputClass']
    outputModel = Param['outputModel']
    varying_pars = Param['varying_pars']
        
    if use_GPU:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    
    # if update_target == 'soft':
    #     copy_step = 1
    #     Param['copy_step'] = copy_step
    
    Param['eps_decay'] = (epsilon - min_eps)/steps_to_min_eps
    eps_decay = Param['eps_decay']
    
    exp_decay_rate = np.exp((np.log(final_lr/learning_rate) * exp_decay_steps)/ N_train)
    Param['exp_decay_rate'] = float(exp_decay_rate)
    
    if PER_b_anneal:
        Param['PER_b_growth'] = (final_PER_b - PER_b)/PER_b_steps
        PER_b_growth = Param['PER_b_growth']
    else:
        Param['PER_b_growth'] = 0.0
        PER_b_growth = Param['PER_b_growth']
        
    if PER_a_anneal:
        Param['PER_a_growth'] = (final_PER_a - PER_a)/PER_a_steps
        PER_a_growth = Param['PER_a_growth']
    else:
        Param['PER_a_growth'] = 0.0
        PER_a_growth = Param['PER_a_growth']

        
    # 2. Generate path for model and tensorboard output, store config ---------------
    savedpath = GeneratePathFolder(outputDir, outputClass, outputModel, varying_pars, N_train, Param)
    saveConfigYaml(Param,savedpath)
    log_dir = os.path.join(savedpath, 'tb')  
    summary_writer = tf.summary.create_file_writer(log_dir)
    if save_ckpt_model:
        os.mkdir(os.path.join(savedpath, 'ckpt'))
    logging.info('Successfully generated path and stored config...')
    
    # 2. Generate data --------------------------------------------------------------
    returns, factors, f_speed = ReturnSampler(N_train, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, seed)
    logging.info('Successfully simulated data...YOU ARE CURRENTLY USING A SEED TO SIMULATE RETURNS. LEAVE IT IF YOU HAVE FOUND A PROPER NN SETTING')
    
    # 3. Instantiate Env --------------------------------------------------------------
    action_space = ActionSpace(KLM)
    env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                    N_train, discount_rate, f_param, f_speed, returns, factors)
    
    if executeRL:
        returns_space = ReturnSpace(RT)
        holding_space = HoldingSpace(KLM)
        QTable = CreateQTable(returns_space,holding_space,action_space,tablr,gamma)
    logging.info('Successfully initialized the market environment...')
    
    # 4. Train DQN algorithm ----------------------------------------------------------
    # instantiate the initial state (return, holding)
    CurrState = env.reset()
    if executeRL:
        env.returns_space = returns_space
        env.holding_space = holding_space
        DiscrCurrState = env.discrete_reset()
    # instantiate the initial state for the benchmark, if required
    if executeGP:
        CurrOptState = env.opt_reset()
        OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
    
    # iteration count to decide when copying weights for the Target Network
    iters = 0
    num_states = len(CurrState)

    TrainNet = DQN(seed, num_states, hidden_units, gamma, start_train, max_experiences, 
                   min_experiences, batch_size, learning_rate, lr_schedule, exp_decay_steps, exp_decay_rate, 
                   action_space, batch_norm_input, batch_norm_hidden, summary_writer, activation, 
                   kernel_initializer, plot_hist, plot_steps_hist, plot_steps,selected_loss, 
                   mom_batch_norm, trainable_batch_norm, DQN_type, use_PER, PER_e,PER_a,PER_b,final_PER_b,PER_b_steps,PER_b_growth,
                   final_PER_a,PER_a_steps,PER_a_growth, clipgrad, clipnorm, clipvalue, clipglob_steps,
                   optimizer_name, optimizer_decay, beta_1, beta_2, eps_opt, update_target,tau, std_rwds, modelname='TrainNet')

    TargetNet = DQN(seed, num_states, hidden_units, gamma, start_train, max_experiences, 
                    min_experiences, batch_size, learning_rate, lr_schedule, exp_decay_steps, exp_decay_rate,
                    action_space, batch_norm_input, batch_norm_hidden, summary_writer, activation,
                    kernel_initializer, plot_hist, plot_steps_hist, plot_steps,selected_loss, 
                    mom_batch_norm, trainable_batch_norm, DQN_type, use_PER, PER_e,PER_a,PER_b,final_PER_b,PER_b_steps,PER_b_growth,
                    final_PER_a,PER_a_steps,PER_a_growth,clipgrad, clipnorm, clipvalue, clipglob_steps, optimizer_name, 
                    optimizer_decay, beta_1, beta_2, eps_opt, update_target, tau, std_rwds, modelname='TargetNet')
    logging.info('Successfully initialized the Deep Q Networks...YOU ARE CURRENTLY USING A SEED TO INITIALIZE WEIGHTS. LEAVE IT IF YOU HAVE FOUND A PROPER NN SETTING')

    
    if Param['do_pretrain']:
        os.mkdir(os.path.join(savedpath, 'ckpt_pt'))
        N_pretrain = Param['N_pretrain']
        lr_schedule = Param['lrate_schedule_pretrain']
        save_ckpt_pretrained_model = Param['save_ckpt_pretrained_model']
        
        pt_returns, pt_factors, pt_f_speed = ReturnSampler(N_pretrain, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, seed)
        pt_env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                           N_pretrain, discount_rate, f_param, pt_f_speed, pt_returns, pt_factors)
        
        PreTrainNet = DQN(seed, num_states, hidden_units, gamma, start_train, max_experiences, 
                           min_experiences, batch_size, learning_rate, lr_schedule, exp_decay_steps, exp_decay_rate, 
                           action_space, batch_norm_input, batch_norm_hidden, summary_writer, activation, 
                           kernel_initializer, plot_hist, plot_steps_hist, plot_steps,selected_loss, 
                           mom_batch_norm, trainable_batch_norm, DQN_type, use_PER, PER_e,PER_a,PER_b,final_PER_b,PER_b_steps,PER_b_growth,
                           final_PER_a,PER_a_steps,PER_a_growth, clipgrad, clipnorm, clipvalue, clipglob_steps,
                           optimizer_name, optimizer_decay, beta_1, beta_2, eps_opt, update_target,tau, std_rwds,
                           modelname='PreTrainNet',pretraining_mode=True)
        
        PreTargetNet = DQN(seed, num_states, hidden_units, gamma, start_train, max_experiences, 
                        min_experiences, batch_size, learning_rate, lr_schedule, exp_decay_steps, exp_decay_rate,
                        action_space, batch_norm_input, batch_norm_hidden, summary_writer, activation,
                        kernel_initializer, plot_hist, plot_steps_hist, plot_steps,selected_loss, 
                        mom_batch_norm, trainable_batch_norm, DQN_type, use_PER, PER_e,PER_a,PER_b,final_PER_b,PER_b_steps,PER_b_growth,
                        final_PER_a,PER_a_steps,PER_a_growth,clipgrad, clipnorm, clipvalue, clipglob_steps, optimizer_name, 
                        optimizer_decay, beta_1, beta_2, eps_opt, update_target, tau, std_rwds,
                        modelname='PreTargetNet',pretraining_mode=True)
        # TODO check here the level of exploration to use
        PreTraining(pt_returns, pt_factors, pt_f_speed, pt_env, PreTrainNet, PreTargetNet, N_pretrain, 
                    min_eps, copy_step, savedpath, save_ckpt_pretrained_model)

        TrainNet.model.load_weights(os.path.join(savedpath,'DQN_pretrained_weights'))
        TargetNet.model.load_weights(os.path.join(savedpath,'DQN_pretrained_weights'))
        
    
    for i in tqdm(iterable=range(N_train + 1), desc='Training DQNetwork'):
        
        if executeDRL:
            epsilon = max(min_eps, epsilon - eps_decay) # linear decay
            shares_traded = TrainNet.eps_greedy_action(CurrState, epsilon)
            NextState, Result, NextFactors = env.step(CurrState, shares_traded, i)
            env.store_results(Result, i)

            exp = {'s': CurrState, 'a': shares_traded, 'r': Result['Reward_DQN'], 's2': NextState, 'f': NextFactors}

            TrainNet.add_experience(exp)
            if i == min_experiences:
                TrainNet.add_test_experience()
                # TrainNet.compute_test_target(TargetNet)
        
            TrainNet.train(TargetNet, i)
            #TrainNet.test(TargetNet, i)
                
            CurrState = NextState
            
            iters += 1
            if (iters % copy_step == 0) and (i > TrainNet.start_train):
                TargetNet.copy_weights(TrainNet)
                #env.res_df.loc[i + 1,'Copy_Target'] = 'Y'
                # TrainNet.compute_test_target(TargetNet)
            if executeGP and (i % pdist_steps == 0) and (i >= TrainNet.min_experiences):
                TrainNet.compute_portfolio_distance(env, OptRate, DiscFactorLoads, i)
                
            if save_ckpt_model and (i % save_ckpt_model == 0):
                TrainNet.model.save_weights(os.path.join(savedpath, 'ckpt','DQN_{}_it_weights'.format(i)), 
                                            save_format='tf')
                 
        if executeRL:
            shares_traded = QTable.chooseAction(DiscrCurrState, epsilon)
            DiscrNextState, Result = env.discrete_step(DiscrCurrState, shares_traded, i)
            env.store_results(Result, i)
            QTable.update(DiscrCurrState,DiscrNextState,shares_traded,Result)
            DiscrCurrState = DiscrNextState
        
        if executeGP:
            NextOptState, OptResult = env.opt_step(CurrOptState, OptRate, DiscFactorLoads, i)
            env.store_results(OptResult, i) 
            CurrOptState = NextOptState
            
            
    logging.info('Successfully trained the Deep Q Network...')
    # 5. Plot and save outputs ----------------------------------------------------------     
    if save_results:
        env.save_outputs(savedpath)
    
    if executeRL and save_table:
        QTable.save(savedpath,N_train)
        logging.info('Successfully plotted and stored results...')
    
    if save_model:
        TrainNet.model.save_weights(os.path.join(savedpath,'DQN_final_weights'), save_format='tf')
        logging.info('Successfully saved DQN weights...')
    
    if out_of_sample_test:

        test_returns, test_factors, test_f_speed = ReturnSampler(N_test, sigmaf, f0, f_param, sigma, plot_inputs, HalfLife, seed)
        test_env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                             N_test, discount_rate, f_param, test_f_speed, test_returns, test_factors)
        
        if executeDRL:
            CurrState = test_env.reset()
        if executeRL:
            test_env.returns_space = returns_space
            test_env.holding_space = holding_space
            DiscrCurrState = test_env.discrete_reset()
        if executeGP:
            CurrOptState = test_env.opt_reset()
            OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()        
        
        for i in tqdm(iterable=range(N_test + 1), desc='Testing DQNetwork'):
            if executeDRL:
                shares_traded = TrainNet.greedy_action(CurrState)
                NextState, Result, NextFactors = test_env.step(CurrState, shares_traded, i)
                test_env.store_results(Result, i)
                CurrState = NextState

            if executeRL:
                shares_traded = int(QTable.chooseGreedyAction(DiscrCurrState))
                DiscrNextState, Result = test_env.discrete_step(DiscrCurrState, shares_traded, i)
                test_env.store_results(Result, i)
                DiscrCurrState = DiscrNextState
            
            if executeGP:
                NextOptState, OptResult = test_env.opt_step(CurrOptState, OptRate, DiscFactorLoads, i)
                test_env.store_results(OptResult, i) 
                CurrOptState = NextOptState
    
        test_env.save_outputs(savedpath, test=True)

if __name__ == "__main__":
    RunTraders(Param)