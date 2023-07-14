# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:27:31 2023

@author: ab978
"""

import os, pdb
import pandas as pd
import numpy as np
import gin
gin.enter_interactive_mode()
from joblib import Parallel, delayed
from utils.test import Out_sample_vs_gp
from utils.env import MarketEnv, CashMarketEnv, ShortCashMarketEnv, MultiAssetCashMarketEnv, ShortMultiAssetCashMarketEnv
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


model = '20230713_real_universal_train_False_split_pct_0.5' #'20230712_real_universal_train_False'
experiment = 'universal_train_False_split_pct_0.5_seed_16' #'universal_train_False_seed_16'

# Load parameters and get the path
query = gin.query_parameter
outputClass = p["outputClass"]
tag = p["algo"]
seed = p["seed"]
modelpath = "outputs/{}/{}".format(outputClass, model)
length = get_exp_length(modelpath)
data_dir = "outputs/{}/{}/{}/{}".format(outputClass, model, length, experiment)
gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)


gin.bind_parameter('load_real_data.universal_train', True)  
# gin.bind_parameter('load_real_data.split_pct', 0.9)  
p['ep_ppo'] = 2500 #'best'

rng = np.random.RandomState(query("%SEED"))

action_space = ActionSpace()

if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
    if query("%TIME_DEPENDENT"):
        input_shape = (len(query('%F_PARAM')) + 2,)
    else:
        input_shape = (len(query('%F_PARAM')) + 1,)
else:
    input_shape = (3,)


train_agent = PPO(
    input_shape=input_shape, action_space=action_space, rng=rng
)


if query('%LOAD_PRETRAINED_PATH'):
    p['ep_ppo'] = None
else:
    p['ep_ppo'] = 'best'

if p['ep_ppo']:
    train_agent.model = load_PPOmodel(data_dir, p['ep_ppo'], model=train_agent.model)
else:
    train_agent.model = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"), model=train_agent.model)

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

for _ in range(10):
    res_df = oos_test.run_test(train_agent, return_output=True)

    print(res_df['Reward_PPO'].cumsum().iloc[-1])