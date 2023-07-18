# -*- coding: utf-8 -*-
import os, pdb
import pandas as pd
import numpy as np
import gin
gin.enter_interactive_mode()
from joblib import Parallel, delayed
from utils.test import Out_sample_vs_gp
from utils.env import MarketEnv,RealMarketEnv,  CashMarketEnv, ShortCashMarketEnv, MultiAssetCashMarketEnv, ShortMultiAssetCashMarketEnv
from agents.PPO import PPO
from utils.spaces import ActionSpace, ResActionSpace
from utils.common import readConfigYaml
from utils.plot import load_PPOmodel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Read config ----------------------------------------------------------------
p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))

def get_exp_length(modelpath):
    # get the latest created folder "length"
    all_subdirs = [
        os.path.join(modelpath, d)
        for d in os.listdir(modelpath)
        if os.path.isdir(os.path.join(modelpath, d))
    ]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    length = os.path.split(latest_subdir)[-1]    
    return length


def parallel_test(seed,test_class,train_agent,data_dir):
    gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
    test_class.rnd_state = seed
    # TODO temp
    # gin.bind_parameter('%T_STUD', True)  
    gin.bind_parameter('%UNIVERSAL_TRAIN', True)
    res_df = test_class.run_test(train_agent, return_output=True)
    return res_df

N = 50
model_to_change = None #'202305120_GP_scratch_pt'
models_experiments =  [('20230717_real_boot_universal_train_False_split_pct_0.8_rho_boot_0.4','universal_train_False_split_pct_0.8_rho_boot_0.4_seed_635')]


                     

for me in models_experiments:
    model = me[0]
    experiment = me[1]
    
    
    # Load parameters and get the path
    query = gin.query_parameter
    outputClass = p["outputClass"]
    tag = p["algo"]
    seed = p["seed"]
    modelpath = "outputs/{}/{}".format(outputClass, model)
    length = get_exp_length(modelpath)
    data_dir = "outputs/{}/{}/{}/{}".format(outputClass, model, length, experiment)
    gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
    p['N_test'] = gin.query_parameter('%LEN_SERIES')
    rng = np.random.RandomState(query("%SEED"))
    
    # Load the elements for producing the plot
    if query("%MV_RES"):
        action_space = ResActionSpace()
    else:
        action_space = ActionSpace()
    
    
    if gin.query_parameter('%MULTIASSET'):
        n_assets = len(gin.query_parameter('%HALFLIFE'))
        n_factors = len(gin.query_parameter('%HALFLIFE')[0])
        inputs = gin.query_parameter('%INPUTS')
        if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
            if 'sigma' in inputs and 'corr' in inputs:
                input_shape = (int(n_factors*n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
            else:
                input_shape = (int(n_factors*n_assets+n_assets+1),1)
        else:
            if 'sigma' in inputs and 'corr' in inputs:
                input_shape = (int(n_factors*n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
            else:
                input_shape = (int(n_factors*n_assets+n_assets+1),1)

    else:
        if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
            if query("%TIME_DEPENDENT"):
                input_shape = (len(query('%F_PARAM')) + 2,)
            else:
                input_shape = (len(query('%F_PARAM')) + 1,)
        else:
            # if query("%RHO_BOOT"):
            #     input_shape = (4,)
            # else:
                # input_shape = (3,)
            input_shape = (3,)
    
    
    train_agent = PPO(
        input_shape=input_shape, action_space=action_space, rng=rng
    )
    
    

    if query('%LOAD_PRETRAINED_PATH'):
        p['ep_ppo'] = None
    else:
        p['ep_ppo'] = 'best'
    p['ep_ppo'] = 3000
    if p['ep_ppo']:
        train_agent.model = load_PPOmodel(data_dir, p['ep_ppo'], model=train_agent.model)
    else:
        train_agent.model = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"), model=train_agent.model)
    
    
    if gin.query_parameter('%MULTIASSET'):
        if 'Short' in str(gin.query_parameter('%ENV_CLS')):
            env = ShortMultiAssetCashMarketEnv
        else:
            env = MultiAssetCashMarketEnv
    else:
        if query('%EXPERIMENT_TYPE') == 'Real':
            env = RealMarketEnv
        else:
            env = MarketEnv
    
    oos_test = Out_sample_vs_gp(
            savedpath=None,
            tag=tag[0],
            experiment_type=query("%EXPERIMENT_TYPE"),
            env_cls=env,
            MV_res=query("%MV_RES"),
            N_test=p['N_test'],
            mv_solution=True
        )

    
    rng_seeds = np.random.RandomState(476)
    seeds = rng_seeds.choice(100000,N)

    # oos_test.rnd_state = 120 
    # res_df = oos_test.run_test(train_agent, return_output=True)
    
    if query('%EXPERIMENT_TYPE') == 'Real':
        pass
        rewards = []
        data  = pd.read_csv('data/{}.csv'.format(gin.query_parameter('load_real_data.datafile')),index_col=0)
        symbols = data.columns.get_level_values(0).unique()[1:]
        for s in symbols:
            gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
            oos_test.rnd_state = 34
            gin.bind_parameter('%UNIVERSAL_TRAIN', True)
            gin.bind_parameter('load_real_data.symbol',s)
            res_df = oos_test.run_test(train_agent, return_output=True)
            print(res_df['Reward_PPO'].cumsum().iloc[-1])
            rewards.append(res_df)
    else:
        rewards = Parallel(n_jobs=p['cores'])(delayed(parallel_test)(
                s, oos_test,train_agent,data_dir) for s in seeds)
    
    if me[0] == model_to_change:
        # create an HDF5 file and store the dataframes in it
        with pd.HDFStore('outputs/full_results/{}_modified.h5'.format(model)) as store:
            for i, df in enumerate(rewards):
                store[f'df_{i+1}'] = df
        # rewards_ppo = pd.concat(list(map(list, zip(*rewards)))[0],axis=1)
        rewards_ppo = pd.concat([df['Reward_PPO'] for df in rewards], ignore_index=True, axis=1)
        rewards_ppo.to_csv('outputs/cumrewards/{}_ppo_modified.csv'.format(model))
        rewards_gp = pd.concat([df['OptReward'] for df in rewards], ignore_index=True, axis=1)
        rewards_gp.to_csv('outputs/cumrewards/{}_gp_modified.csv'.format(model))
        rewards_mv = pd.concat([df['MVReward'] for df in rewards], ignore_index=True, axis=1)
        rewards_mv.to_csv('outputs/cumrewards/{}_mv_modified.csv'.format(model))
    else:
        # create an HDF5 file and store the dataframes in it
        with pd.HDFStore('outputs/full_results/{}.h5'.format(model)) as store:
            for i, df in enumerate(rewards):
                store[f'df_{i+1}'] = df
        # rewards_ppo = pd.concat(list(map(list, zip(*rewards)))[0],axis=1)
        rewards_ppo = pd.concat([df['Reward_PPO'] for df in rewards], ignore_index=True, axis=1)
        rewards_ppo.to_csv('outputs/cumrewards/{}_ppo.csv'.format(model))
        rewards_gp = pd.concat([df['OptReward'] for df in rewards], ignore_index=True, axis=1)
        rewards_gp.to_csv('outputs/cumrewards/{}_gp.csv'.format(model))
        rewards_mv = pd.concat([df['MVReward'] for df in rewards], ignore_index=True, axis=1)
        rewards_mv.to_csv('outputs/cumrewards/{}_mv.csv'.format(model))
