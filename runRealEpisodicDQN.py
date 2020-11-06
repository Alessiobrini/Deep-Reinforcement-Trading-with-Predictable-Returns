# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:09:28 2020

@author: aless
"""
# 0. importing section initialize logger.--------------------------------------
import logging, os, pdb
from utils.readYaml import readConfigYaml, saveConfigYaml 
from utils.generateLogger import generate_logger
from utils.SavePath import GeneratePathFolder
from utils.SimulateData import ReturnSampler, create_lstm_tensor
from utils.MarketEnv import MarketEnv, RecurrentMarketEnv, ActionSpace, ReturnSpace, HoldingSpace, CreateQTable
from utils.DQN import DQN
from utils.Regressions import CalculateLaggedSharpeRatio, RunModels
from utils.Out_of_sample_testing import Out_sample_real_test
# from utils.LaunchIpynbs import runNBs
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np

# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ---------------------------------------------------------------- 
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramRealDQN.yaml'))
logging.info('Successfully read config file with parameters...')

def RunRealDQNTraders(Param):
    
    # 0. EXTRACT PARAMETERS ----------------------------------------------------------
    epsilon = Param['epsilon']
    min_eps = Param['min_eps']
    optimal_expl = Param['optimal_expl']
    alpha = Param['alpha']
    gamma = Param['gamma']
    kappa = Param['kappa']
    std_rwds = Param['std_rwds']
    DQN_type = Param['DQN_type']
    recurrent_env = Param['recurrent_env']
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
    clipgrad = Param['clipgrad']
    clipnorm = Param['clipnorm']
    clipvalue = Param['clipvalue']
    clipglob_steps = Param['clipglob_steps']
    optimizer_name = Param['optimizer_name']
    beta_1 = Param['beta_1']
    beta_2 = Param['beta_2']
    eps_opt = Param['eps_opt']
    hidden_units = Param['hidden_units']
    batch_size = Param['batch_size']
    copy_step = Param['copy_step']
    update_target = Param['update_target']
    tau = Param['tau']
    learning_rate = Param['learning_rate']
    lr_schedule = Param['lr_schedule']
    exp_decay_steps = Param['exp_decay_steps']
    exp_decay_rate = Param['exp_decay_rate']
    hidden_memory_units = Param['hidden_memory_units']
    unfolding = Param['unfolding']
    KLM = Param['KLM']
    zero_action = Param['zero_action']
    RT = Param['RT']
    tablr = Param['tablr']
    # Data Simulation
    symbol = Param['symbol']
    factor_lb = Param['factor_lb']
    CostMultiplier = Param['CostMultiplier']
    discount_rate = Param['discount_rate']
    Startholding = Param['Startholding']
    # Experiment and storage
    start_train = Param['start_train']
    seed_init = Param['seed_init']
    out_of_sample_test = Param['out_of_sample_test']
    executeDRL = Param['executeDRL']
    executeRL = Param['executeRL']
    executeGP = Param['executeGP']
    executeMV = Param['executeMV']
    save_results = Param['save_results']
    save_table = Param['save_table']
    plot_hist = Param['plot_hist']
    plot_steps_hist = Param['plot_steps_hist']
    plot_steps = Param['plot_steps']
    save_model = Param['save_model']
    save_ckpt_model = Param['save_ckpt_model']
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
    
    if not recurrent_env:
        Param['unfolding'] = unfolding = 1 
    
    if update_target == 'soft':
        if Param['varying_type'] == 'random_search':
            copy_step = Param['copy_step'] = 1
        else:
            assert copy_step == 1, 'Soft target updates require copy step to be 1'
        
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
                
    
    # 1. GET REAL DATA AND MAKE REGRESSIONS --------------------------------------------------------------
    # import and rearrange data
    asset_df = pd.read_parquet('data/daily_futures/daily_bars_{}_1990-07-01_2020-07-15.parquet.gzip'.format(symbol))
    asset_df.set_index('date', inplace=True)
    asset_df = asset_df.iloc[::-1]
    price_series = asset_df['close_p']
    # calculate predicting factors
    df = CalculateLaggedSharpeRatio(price_series,factor_lb,symbol)
    y, X = df[df.columns[0]], df[df.columns[1:]]  
    # do regressions
    params_retmodel, params_meanrev, fitted_retmodel, fitted_ous = RunModels(y,X)
    # get results
    dates = df.index
    returns = df.iloc[:,0].values
    factors = df.iloc[:,1:].values
    sigma = df.iloc[:,0].std()
    f_param = params_retmodel['params']
    f_speed = np.array([*params_meanrev.values()]).ravel()
    HalfLife = -np.around(np.log(2)/f_speed,2)
    
    N_train = len(returns)
    Param['N_train'] = N_train
    Param['HalfLife'] = HalfLife
    Param['f_speed'] = f_speed
    Param['f_param'] = f_param
    Param['sigma'] = sigma
    steps_to_min_eps = int(round(df.shape[0]*0.8,-1)) #TODO change hard coded now
    Param['steps_to_min_eps'] = steps_to_min_eps
    eps_decay = (epsilon - min_eps)/steps_to_min_eps
    Param['eps_decay'] = eps_decay
    max_experiences = int(round(df.shape[0]/4,-1))
    Param['max_experiences'] = max_experiences
    
    if save_ckpt_model:
        save_ckpt_steps = int((len(returns)-1)/save_ckpt_model)

    if recurrent_env:
        returns_tens = create_lstm_tensor(returns.reshape(-1,1), unfolding)
        factors_tens = create_lstm_tensor(factors, unfolding)
    logging.info('Successfully loaded data...')
    
    # 2. PATH FOR MODEL (CKPT) AND TB OUTPUT, STORE CONFIG ---------------    
    savedpath = GeneratePathFolder(outputDir, outputClass, outputModel, varying_pars, N_train, Param)
    saveConfigYaml(Param,savedpath)
    log_dir = os.path.join(savedpath, 'tb')  
    summary_writer = tf.summary.create_file_writer(log_dir)
    if save_ckpt_model and not os.path.exists(os.path.join(savedpath, 'ckpt')):
        os.makedirs(os.path.join(savedpath, 'ckpt'))
    elif save_ckpt_model and os.path.exists(os.path.join(savedpath, 'ckpt')):
        pass
    logging.info('Successfully generated path and stored config...')
    
    # 3. CREATE MARKET ENVIRONMENTS --------------------------------------------------------------
    # market env for DQN or its variant
    action_space = ActionSpace(KLM, zero_action)
    if recurrent_env:
        env = RecurrentMarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                        N_train, discount_rate, f_param, f_speed, returns, factors, returns_tens, factors_tens, dates=dates)
    else:
        env = MarketEnv(HalfLife, Startholding, sigma, CostMultiplier, kappa, 
                        N_train, discount_rate, f_param, f_speed, returns, factors, dates=dates)
    # market env for tab Q learning
    if executeRL:
        returns_space = ReturnSpace(RT)
        holding_space = HoldingSpace(KLM)
        QTable = CreateQTable(returns_space,holding_space,action_space,tablr,gamma)
    logging.info('Successfully initialized the market environment...')
    
    # 4. CREATE INITIAL STATE AND NETWORKS ----------------------------------------------------------
    # instantiate the initial state (return, holding) for DQN
    CurrState, CurrFactors = env.reset()
    # instantiate the initial state (return, holding) for TabQ
    if executeRL:
        env.returns_space = returns_space
        env.holding_space = holding_space
        DiscrCurrState = env.discrete_reset()
    # instantiate the initial state for the benchmark
    if executeGP:
        CurrOptState = env.opt_reset()
        OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
    # instantiate the initial state for the markovitz solution
    if executeMV:
        CurrMVState = env.opt_reset()

    input_shape = CurrState.shape
    # create train and target network
    TrainQNet = DQN(seed_init,DQN_type,recurrent_env,gamma,max_experiences, update_target,tau,input_shape, 
                    hidden_units, hidden_memory_units, batch_size, selected_loss,learning_rate, start_train, optimizer_name,batch_norm_input,
                    batch_norm_hidden, activation, kernel_initializer, plot_hist, plot_steps_hist, plot_steps,
                    summary_writer, action_space,  use_PER,  PER_e, PER_a, PER_b, final_PER_b, PER_b_steps, 
                    PER_b_growth, final_PER_a,PER_a_steps,PER_a_growth, clipgrad, clipnorm, clipvalue, 
                    clipglob_steps, beta_1, beta_2, eps_opt, std_rwds,lr_schedule, exp_decay_steps, 
                    exp_decay_rate,modelname='TrainQNet', stop_train=steps_to_min_eps)
    TargetQNet = DQN(seed_init,DQN_type,recurrent_env,gamma,max_experiences, update_target,tau,input_shape, 
                    hidden_units, hidden_memory_units, batch_size, selected_loss,learning_rate, start_train, optimizer_name,batch_norm_input,
                    batch_norm_hidden, activation, kernel_initializer, plot_hist, plot_steps_hist, plot_steps,
                    summary_writer, action_space,  use_PER,  PER_e, PER_a, PER_b, final_PER_b, PER_b_steps, 
                    PER_b_growth, final_PER_a,PER_a_steps,PER_a_growth, clipgrad, clipnorm, clipvalue, 
                    clipglob_steps, beta_1, beta_2, eps_opt, std_rwds,lr_schedule, exp_decay_steps, 
                    exp_decay_rate,modelname='TargetQNet', stop_train=steps_to_min_eps)

    logging.info('Successfully initialized the Deep Q Networks...YOU ARE CURRENTLY USING A SEED TO INITIALIZE WEIGHTS.')        
    # 5. TRAIN ALGORITHM ----------------------------------------------------------
    iters = 0
    for i in tqdm(iterable=range(len(returns)-1), desc='Training DQNetwork'):
        if executeDRL:
            epsilon = max(min_eps, epsilon - eps_decay)
            if not optimal_expl:
                shares_traded = TrainQNet.eps_greedy_action(CurrState, epsilon)
            else:
                OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
                shares_traded = TrainQNet.alpha_beta_greedy_action(CurrState,CurrFactors, epsilon,
                                                                   OptRate, DiscFactorLoads, alpha, env)
            
            NextState, Result, NextFactors = env.step(CurrState, shares_traded, i)
            env.store_results(Result, i)

            exp = {'s': CurrState, 'a': shares_traded, 'r': Result['Reward_DQN'], 's2': NextState, 'f': NextFactors}
            TrainQNet.add_experience(exp)       
            TrainQNet.train(TargetQNet, i)
            
            CurrState = NextState
            CurrFactors = NextFactors
            iters += 1
            if (iters % copy_step == 0) and (i > TrainQNet.start_train):
                TargetQNet.copy_weights(TrainQNet)

            if save_ckpt_model and (i % save_ckpt_steps == 0) and (i > TrainQNet.start_train):
                TrainQNet.model.save_weights(os.path.join(savedpath, 'ckpt','DQN_{}_it_weights'.format(i)), 
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
            
        if executeMV:
            NextMVState, MVResult = env.mv_step(CurrMVState, i)
            env.store_results(MVResult, i) 
            CurrMVState = NextMVState
            
        # 5.1 OUT OF SAMPLE TEST ----------------------------------------------------------   
        if out_of_sample_test:
            if (i % save_ckpt_steps == 0) and (i != 0) and (i > TrainQNet.start_train):
                if not executeRL:
                    QTable = None
                Out_sample_real_test(N_train, returns, factors, f_speed, f_param, sigma, HalfLife, 
                                Startholding,CostMultiplier,kappa,discount_rate,executeDRL, 
                                executeRL,executeMV,RT,KLM,executeGP,TrainQNet,savedpath,i, 
                                recurrent_env,unfolding,QTable)
                                
    logging.info('Successfully trained the Deep Q Network...')
    # 6. STORE RESULTS ----------------------------------------------------------     
    if save_results:
        env.save_outputs(savedpath)
    
    if executeRL and save_table:
        QTable.save(savedpath,N_train)
        logging.info('Successfully plotted and stored results...')
    
    if save_model:
        TrainQNet.model.save_weights(os.path.join(savedpath,'DQN_final_weights'), save_format='tf')
        logging.info('Successfully saved DQN weights...')


if __name__ == "__main__":
    RunRealDQNTraders(Param)