# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:34:30 2020

@author: alessiobrini
"""

import os

if any("SPYDER" in name for name in os.environ):
    from IPython import get_ipython

    get_ipython().magic("reset -sf")

import os, logging, sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec
import tensorflow as tf
from scipy.stats import ks_2samp,kstest,ttest_ind
import pdb
import glob
import seaborn as sns
sns.set_style("darkgrid")
import gin
gin.enter_interactive_mode()
from joblib import Parallel, delayed

from utils.plot import (
    plot_pct_metrics,
    plot_abs_metrics,
    plot_BestActions,
    plot_vf,
    load_DQNmodel,
    load_PPOmodel,
    plot_portfolio,
    plot_action,
    plot_costs,
    plot_2asset_holding
)
from utils.test import Out_sample_vs_gp
from utils.env import MarketEnv, CashMarketEnv, ShortCashMarketEnv, MultiAssetCashMarketEnv, ShortMultiAssetCashMarketEnv
from agents.DQN import DQN
from agents.PPO import PPO
from utils.spaces import ActionSpace, ResActionSpace
from utils.common import readConfigYaml, generate_logger, format_tousands, set_size

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def runplot_metrics(p):

    N_test = p["N_test"]
    outputClass = p["outputClass"]
    tag = p["algo"]
    if 'DQN' in tag:
        hp = p["hyperparams_model_dqn"]
        outputModels = p["outputModels_dqn"]
    elif 'PPO' in tag:
        hp = p["hyperparams_model_ppo"]
        outputModels = p["outputModels_ppo"]


    if hp is not None:
        outputModel = [exp.format(*hp) for exp in outputModels]
    else:
        outputModel = outputModels

    colors = ["blue", "y", "green", "black"]
    # colors = []
    random.seed(2212)  # 7156
    for _ in range(len(outputModel)):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        colors.append(color)

    for t in tag:

        
        var_plot = [v.format(format_tousands(N_test), t) for v in p['var_plots'] ]

        for it, v in enumerate(var_plot):
            # pdb.set_trace()
            if not "Abs" in v and not 'Pdist' in v:
                # read main folder
                fig = plt.figure(figsize=set_size(width=1000)) #600
                # fig.subplots_adjust(wspace=0.2, hspace=0.6)
                ax = fig.add_subplot()
            for k, out_mode in enumerate(outputModel):
                if "Abs" in v or 'Pdist' in v:
                    # read main folder
                    fig = plt.figure(figsize=set_size(width=1000)) #600
                    # fig.subplots_adjust(wspace=0.2, hspace=0.6)
                    ax = fig.add_subplot()
                modelpath = "outputs/{}/{}".format(outputClass, out_mode)

                # get the latest created folder "length"
                all_subdirs = [
                    os.path.join(modelpath, d)
                    for d in os.listdir(modelpath)
                    if os.path.isdir(os.path.join(modelpath, d))
                ]
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                length = os.path.split(latest_subdir)[-1]

                data_dir = "outputs/{}/{}/{}".format(outputClass, out_mode, length)

                # Recover and plot generated multi test OOS ----------------------------------------------------------------
                filtered_dir = [
                    dirname
                    for dirname in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(os.getcwd(), data_dir, dirname))
                ]
                logging.info(
                    "Plotting experiment {} for variable {}...".format(out_mode, v)
                )
                dfs = []
                for exp in filtered_dir:
                    exp_path = os.path.join(data_dir, exp)
                    df = pd.read_parquet(os.path.join(exp_path, v))

                    # filenamep = os.path.join(
                    #     data_dir, exp, "config.yaml".format(length)
                    # )
                    # p_mod = readConfigYaml(filenamep)
                    filenamep = os.path.join(data_dir, exp, "config.gin")

                    dfs.append(df)

                dataframe = pd.concat(dfs)
                dataframe.index = range(len(dfs))

                # pdb.set_trace()
                if 'PPO' in tag and p['ep_ppo']:
                    dataframe = dataframe.iloc[:,:dataframe.columns.get_loc(p['ep_ppo'])]

                if "Abs" in v or 'Pdist' in v:

                    dfs_opt = []
                    for exp in filtered_dir:
                        exp_path = os.path.join(data_dir, exp)
                        df_opt = pd.read_parquet(
                            os.path.join(exp_path, v.replace(t, "GP"))
                        )
                        dfs_opt.append(df_opt)
                    dataframe_opt = pd.concat(dfs_opt)
                    dataframe_opt.index = range(len(dfs_opt))
                    if 'PPO' in tag and p['ep_ppo']:
                        dataframe_opt = dataframe_opt.iloc[:,:dataframe_opt.columns.get_loc(p['ep_ppo'])]
                    # pdb.set_trace()
                    plot_abs_metrics(
                        ax,
                        dataframe,
                        dataframe_opt,
                        data_dir,
                        N_test,
                        v,
                        colors=colors[k],
                        i=it,
                    )                    
                    
                    if 'Pdist' in v:
                        std = 1e+10
                        ax.set_ylim(0.0, 0.0 + std)
                    else:
                        import math
                        value = dataframe_opt.iloc[0, 2]
                        odm = math.floor(math.log(value, 10))
                        ax.set_ylim(value - (10**(odm)), value + 0.5*(10**(odm)))


                else:
                    plot_pct_metrics(
                        ax,
                        dataframe,
                        data_dir,
                        N_test,
                        v,
                        colors=colors[k],
                        params_path=filenamep,
                    )
                    ax.set_ylim(-20, 150)
                logging.info("Plot saved successfully...")



def runplot_value(p):

    outputClass = p["outputClass"]
    tag = p["algo"]
    seed = p["seed"]
    
    if 'DQN' in tag:
        hp_exp = p["hyperparams_exp_dqn"]
        outputModel = p["outputModel_dqn"]
        experiment = p["experiment_dqn"]
    elif 'PPO' in tag:
        hp_exp = p["hyperparams_exp_ppo"]
        outputModel = p["outputModel_ppo"]
        experiment = p["experiment_ppo"]

    if hp_exp:
        outputModel = outputModel.format(*hp_exp)
        experiment = experiment.format(*hp_exp, seed)

    modelpath = "outputs/{}/{}".format(outputClass, outputModel)
    # get the latest created folder "length"
    all_subdirs = [
        os.path.join(modelpath, d)
        for d in os.listdir(modelpath)
        if os.path.isdir(os.path.join(modelpath, d))
    ]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    length = os.path.split(latest_subdir)[-1]
    data_dir = "outputs/{}/{}/{}/{}".format(
        outputClass, outputModel, length, experiment
    )

    gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)

    fig = plt.figure(figsize=set_size(width=1000.0))
    ax = fig.add_subplot()

    if "DQN" in tag:
        if p['n_dqn']:
            model, actions = load_DQNmodel(data_dir, p['n_dqn'])
        else:
            model, actions = load_DQNmodel(data_dir, gin.query_parameter("%N_TRAIN")) 
        plot_vf(model, actions, p['holding'], ax=ax, optimal=p['optimal'])

        ax.set_xlabel("y")
        ax.set_ylabel("action-value function")
        ax.legend(ncol=3)

    elif "PPO" in tag:
        if p['ep_ppo']:
            model, actions = load_PPOmodel(data_dir, p['ep_ppo'])
        else:
            model, actions = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"))

        plot_vf(model, actions, p['holding'], ax=ax, optimal=p['optimal'])

        ax.set_xlabel("y")
        ax.set_ylabel("value function")
    else:
        print("Choose proper algorithm.")
        sys.exit()

    # start, end, stepsize = -750, 1250, 500
    # ax1.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
    # start, end, stepsize = -20000, 20000, 20000
    # ax2.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # plt.savefig(
    #     os.path.join("outputs", "figs", "DQN_vf_{}.pdf".format(length)), dpi=300
    # )


def runplot_policy(p):

    outputClass = p["outputClass"]
    tag = p["algo"]
    seed = p["seed"]
    
    if 'DQN' in tag:
        hp_exp = p["hyperparams_exp_dqn"]
        outputModel = p["outputModel_dqn"]
        experiment = p["experiment_dqn"]
    elif 'PPO' in tag:
        hp_exp = p["hyperparams_exp_ppo"]
        outputModel = p["outputModel_ppo"]
        experiment = p["experiment_ppo"]

    if hp_exp:
        outputModel = outputModel.format(*hp_exp)
        experiment = experiment.format(*hp_exp, seed)

    modelpath = "outputs/{}/{}".format(outputClass, outputModel)
    # get the latest created folder "length"
    all_subdirs = [
        os.path.join(modelpath, d)
        for d in os.listdir(modelpath)
        if os.path.isdir(os.path.join(modelpath, d))
    ]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    length = os.path.split(latest_subdir)[-1]
    data_dir = "outputs/{}/{}/{}/{}".format(
        outputClass, outputModel, length, experiment
    )

    fig = plt.figure(figsize=set_size(width=1000.0))
    ax = fig.add_subplot()

    if "DQN" in tag:
        gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
        if p['n_dqn']:
            model, actions = load_DQNmodel(data_dir, p['n_dqn'])
        else:
            model, actions = load_DQNmodel(data_dir, gin.query_parameter("%N_TRAIN"))      
        
        plot_BestActions(model, p['holding'], ax=ax, optimal=p['optimal'])

        ax.set_xlabel("y")
        ax.set_ylabel("best $\mathregular{A_{t}}$")
        ax.legend()

    elif "PPO" in tag:
        gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
        if p['ep_ppo']:
            model, actions = load_PPOmodel(data_dir, p['ep_ppo'])
        else:
            model, actions = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"))
        
        plot_BestActions(model, p['holding'], ax=ax, optimal=p['optimal'])

        ax.set_xlabel("y")
        ax.set_ylabel("best $\mathregular{A_{t}}$")
        ax.legend()
    else:
        print("Choose proper algorithm.")
        sys.exit()

    # start, end, stepsize = -750, 1250, 500
    # ax1.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
    # start, end, stepsize = -20000, 20000, 20000
    # ax2.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # plt.savefig(
    #     os.path.join("outputs", "figs", "DQN_vf_{}.pdf".format(length)), dpi=300
    # )


def parallel_test(seed,test_class,train_agent,data_dir,fullpath=False):
    gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
    # change reward function in order to evaluate in the same way
    if gin.query_parameter('%REWARD_TYPE') == 'cara':
        gin.bind_parameter('%REWARD_TYPE', 'mean_var')
    test_class.rnd_state = seed
    res_df = test_class.run_test(train_agent, return_output=True)
    if fullpath:
        return res_df['Reward_PPO'],res_df['OptReward']
    else:
        return res_df['Reward_PPO'].cumsum().values[-1],res_df['OptReward'].cumsum().values[-1]
    
def parallel_test_wealth(seed,test_class,train_agent,data_dir,fullpath=False):
    gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
    # change reward function in order to evaluate in the same way
    if gin.query_parameter('%REWARD_TYPE') == 'cara':
        gin.bind_parameter('%REWARD_TYPE', 'mean_var')
    test_class.rnd_state = seed
    res_df = test_class.run_test(train_agent, return_output=True)
    if fullpath:
        return res_df['Wealth_PPO'],res_df['OptWealth']
    else:
        return res_df['Wealth_PPO'].values[-1],res_df['OptWealth'].values[-1]


def runplot_distribution(p):

    query = gin.query_parameter
    outputClass = p["outputClass"]
    tag = p["algo"]
    seed = p["seed"]
    
    if 'DQN' in tag:
        hp_exp = p["hyperparams_exp_dqn"]
        outputModel = p["outputModel_dqn"]
        experiment = p["experiment_dqn"]
    elif 'PPO' in tag:
        hp_exp = p["hyperparams_exp_ppo"]
        outputModel = p["outputModel_ppo"]
        experiment = p["experiment_ppo"]

    if hp_exp:
        outputModel = outputModel.format(*hp_exp)
        experiment = experiment.format(*hp_exp, seed)


    modelpath = "outputs/{}/{}".format(outputClass, outputModel)
    # get the latest created folder "length"
    all_subdirs = [
        os.path.join(modelpath, d)
        for d in os.listdir(modelpath)
        if os.path.isdir(os.path.join(modelpath, d))
    ]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    length = os.path.split(latest_subdir)[-1]
    data_dir = "outputs/{}/{}/{}/{}".format(
        outputClass, outputModel, length, experiment
    )


    gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
    gin.bind_parameter('alpha_term_structure_sampler.generate_plot', p['generate_plot'])
    p['N_test'] = gin.query_parameter('%LEN_SERIES')
    
    # gin.bind_parameter('Out_sample_vs_gp.rnd_state',p['random_state'])
    rng = np.random.RandomState(query("%SEED"))

    if query("%MV_RES"):
        action_space = ResActionSpace()
    else:
        action_space = ActionSpace()

    if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
        input_shape = (len(query('%F_PARAM')) + 1,)
    else:
        input_shape = (2,)


    if "DQN" in tag:
        train_agent = DQN(
            input_shape=input_shape, action_space=action_space, rng=rng
        )
        if p['n_dqn']:
            train_agent.model = load_DQNmodel(
                data_dir, p['n_dqn'], model=train_agent.model
            )
        else:
            train_agent.model = load_DQNmodel(
                    data_dir, query("%N_TRAIN"), model=train_agent.model
                )

    elif "PPO" in tag:
        train_agent = PPO(
            input_shape=input_shape, action_space=action_space, rng=rng
        )

        if p['ep_ppo']:
            train_agent.model = load_PPOmodel(data_dir, p['ep_ppo'], model=train_agent.model)
        else:
            train_agent.model = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"), model=train_agent.model)
    else:
        print("Choose proper algorithm.")
        sys.exit()
    
    
    if gin.query_parameter('%MULTIASSET'):
        if 'Short' in str(gin.query_parameter('%ENV_CLS')):
            env = ShortMultiAssetCashMarketEnv
        else:
            env = MultiAssetCashMarketEnv
    else:
        env = MarketEnv
        
    oos_test = Out_sample_vs_gp(
        savedpath=None,
        tag=tag[0],
        experiment_type=query("%EXPERIMENT_TYPE"),
        env_cls=env,
        MV_res=query("%MV_RES"),
        N_test=p['N_test']
    )


    rng_seeds = np.random.RandomState(14)
    seeds = rng_seeds.choice(1000,p['n_seeds'])
    if p['disttoplot'] == 'r':
        title = 'reward'
        rewards = Parallel(n_jobs=p['cores'])(delayed(parallel_test)(
                s, oos_test,train_agent,data_dir,p['fullpath']) for s in seeds)
    elif p['disttoplot'] == 'w':
        title= 'wealth'
        rewards = Parallel(n_jobs=p['cores'])(delayed(parallel_test_wealth)(
                s, oos_test,train_agent,data_dir,p['fullpath']) for s in seeds)

    if p['fullpath']:

        rewards_ppo = pd.concat(list(map(list, zip(*rewards)))[0],axis=1).cumsum()
        rewards_gp = pd.concat(list(map(list, zip(*rewards)))[1],axis=1).cumsum()
        fig = plt.figure(figsize=set_size(width=1000.0))
        ax = fig.add_subplot()
        # rewards.loc[:,:].cumsum().plot(ax=ax)
        # rewards.loc[:len(rewards)//2,:].cumsum().plot(ax=ax)
        rewards_ppo.mean(axis=1).plot(ax=ax, label='ppo_mean')
        # ci = 2 * rewards_ppo.std(axis=1)
        # ax.fill_between(list(rewards_ppo.index),(rewards_ppo.mean(axis=1) - ci), (rewards_ppo.mean(axis=1) + ci), color='tab:blue', alpha=0.5)
        rewards_gp.mean(axis=1).plot(ax=ax, label='gp_mean')
        # ci = 2 * rewards_gp.std(axis=1)
        # ax.fill_between(list(rewards_ppo.index),(rewards_gp.mean(axis=1) - ci), (rewards_gp.mean(axis=1) + ci), color='tab:orange', alpha=0.5)
        ax.set_xlabel("Cumulative reward")
        ax.set_ylabel("Frequency")
        ax.legend()

        # fig = plt.figure(figsize=set_size(width=1000.0))
        # ax = fig.add_subplot()
        # rewards.loc[:,:].plot(ax=ax)
        # rewards.loc[:len(rewards)//2,:].plot(ax=ax)
        # ax.set_xlabel("Cumulative reward")
        # ax.set_ylabel("Frequency")
        # ax.legend()
    else:
        rewards = pd.DataFrame(data=np.array(rewards), columns=['ppo','gp'])

        fig = plt.figure(figsize=set_size(width=1000.0))
        ax = fig.add_subplot()
        rewards['gp'].plot(ax=ax, kind='hist', alpha=0.7,color='tab:orange',bins=50)
        rewards['ppo'].plot(ax=ax, kind='hist', color='tab:blue',bins=50)
        ax.set_xlabel("Cumulative {}".format(title))
        ax.set_ylabel("Frequency")
        ax.legend()
        means, stds = rewards.mean().values, rewards.std().values
        ax.set_title('{} Obs \n Means: PPO {:.2f} GP {:.2f} \n Stds: PPO {:.2f} GP {:.2f}'.format(len(rewards),*means,*stds))
        fig.savefig(os.path.join(data_dir, "cum{}_hist_{}.png".format(title,p['n_seeds'])), dpi=300)
    
        fig = plt.figure(figsize=set_size(width=1000.0))
        ax = fig.add_subplot()
        sns.kdeplot(rewards['gp'].values, bw_method=0.2,ax=ax,color='tab:orange')
        sns.kdeplot(rewards['ppo'].values, bw_method=0.2,ax=ax,color='tab:blue')
        ax.set_xlabel("Cumulative {}".format(title))
        ax.set_ylabel("KDE")
        ax.legend()
        means, stds = rewards.mean().values, rewards.std().values
        ax.set_title('{} Obs \n Means: PPO {:.2f} GP {:.2f} \n Stds: PPO {:.2f} GP {:.2f}'.format(len(rewards),*means,*stds))
        ax.legend(labels=['gp','ppo'],loc=2)
    
        KS, p_V = ks_2samp(rewards.values[:,0], rewards.values[:,1])
        t, p_t = ttest_ind(rewards.values[:,0], rewards.values[:,1])
        ks_text = AnchoredText("Ks Test: pvalue {:.2f} \n T Test: pvalue {:.2f}".format(p_V,p_t),loc=1,prop=dict(size=10))
        ax.add_artist(ks_text)
        fig.savefig(os.path.join(data_dir, "cum{}_density_{}.png".format(title,p['n_seeds'])), dpi=300)

    


def runmultiplot_distribution(p):

    query = gin.query_parameter
    modeltag = p['modeltag']
    outputClass = p["outputClass"]
    tag = p["algo"]
    seed = p["seed"]
    
    folders = [p for p in glob.glob(os.path.join('outputs',outputClass,'{}*'.format(modeltag)))]
    length = os.listdir(folders[0])[0]

    for main_f in folders:    
        length = os.listdir(main_f)[0]
        for f in os.listdir(os.path.join(main_f,length)):
            if 'seed_{}'.format(p['seed']) in f:
                data_dir = os.path.join(main_f,length,f)
                # pdb.set_trace()
                
                gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
                gin.bind_parameter('alpha_term_structure_sampler.generate_plot', p['generate_plot'])
                # change reward function in order to evaluate in the same way
                if query('%REWARD_TYPE') == 'cara':
                    gin.bind_parameter('%REWARD_TYPE', 'mean_var')
                p['N_test'] = gin.query_parameter('%LEN_SERIES')
                
                # gin.bind_parameter('Out_sample_vs_gp.rnd_state',p['random_state'])
                rng = np.random.RandomState(query("%SEED"))
            
                if query("%MV_RES"):
                    action_space = ResActionSpace()
                else:
                    action_space = ActionSpace()
                    
                if gin.query_parameter('%MULTIASSET'):
                    n_assets = len(gin.query_parameter('%HALFLIFE'))
                    n_factors = len(gin.query_parameter('%HALFLIFE')[0])
                    if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                        # input_shape = (n_factors*n_assets+n_assets+1,1)
                        input_shape = (int(n_factors*n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
                    else:
                        input_shape = (n_assets+n_assets+1,1)
                else:
                    if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                        input_shape = (len(query('%F_PARAM')) + 1,)
                    else:
                        input_shape = (2,)
            
            
                if "DQN" in tag:
                    train_agent = DQN(
                        input_shape=input_shape, action_space=action_space, rng=rng
                    )
                    if p['n_dqn']:
                        train_agent.model = load_DQNmodel(
                            data_dir, p['n_dqn'], model=train_agent.model
                        )
                    else:
                        train_agent.model = load_DQNmodel(
                                data_dir, query("%N_TRAIN"), model=train_agent.model
                            )
            
                elif "PPO" in tag:
                    train_agent = PPO(
                        input_shape=input_shape, action_space=action_space, rng=rng
                    )
            
                    if p['ep_ppo']:
                        train_agent.model = load_PPOmodel(data_dir, p['ep_ppo'], model=train_agent.model)
                    else:
                        train_agent.model = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"), model=train_agent.model)
                else:
                    print("Choose proper algorithm.")
                    sys.exit()
                    
                if gin.query_parameter('%MULTIASSET'):
                    if 'Short' in str(gin.query_parameter('%ENV_CLS')):
                        env = ShortMultiAssetCashMarketEnv
                    else:
                        env = MultiAssetCashMarketEnv
                else:
                    env = MarketEnv
                oos_test = Out_sample_vs_gp(
                    savedpath=None,
                    tag=tag[0],
                    experiment_type=query("%EXPERIMENT_TYPE"),
                    env_cls=env,
                    MV_res=query("%MV_RES"),
                    N_test=p['N_test']
                )
                
            
                rng_seeds = np.random.RandomState(14)
                seeds = rng_seeds.choice(1000,p['n_seeds'])
                rewards = Parallel(n_jobs=p['cores'])(delayed(parallel_test)(
                        s, oos_test,train_agent,data_dir,p['fullpath']) for s in seeds)
            
                if p['fullpath']:
            
                    rewards_ppo = pd.concat(list(map(list, zip(*rewards)))[0],axis=1).cumsum()
                    rewards_gp = pd.concat(list(map(list, zip(*rewards)))[1],axis=1).cumsum()
                    fig = plt.figure(figsize=set_size(width=1000.0))
                    ax = fig.add_subplot()
                    # rewards.loc[:,:].cumsum().plot(ax=ax)
                    # rewards.loc[:len(rewards)//2,:].cumsum().plot(ax=ax)
                    rewards_ppo.mean(axis=1).plot(ax=ax, label='ppo_mean')
                    # ci = 2 * rewards_ppo.std(axis=1)
                    # ax.fill_between(list(rewards_ppo.index),(rewards_ppo.mean(axis=1) - ci), (rewards_ppo.mean(axis=1) + ci), color='tab:blue', alpha=0.5)
                    rewards_gp.mean(axis=1).plot(ax=ax, label='gp_mean')
                    # ci = 2 * rewards_gp.std(axis=1)
                    # ax.fill_between(list(rewards_ppo.index),(rewards_gp.mean(axis=1) - ci), (rewards_gp.mean(axis=1) + ci), color='tab:orange', alpha=0.5)
                    ax.set_xlabel("Cumulative reward")
                    ax.set_ylabel("Frequency")
                    ax.legend()
            
                    # fig = plt.figure(figsize=set_size(width=1000.0))
                    # ax = fig.add_subplot()
                    # rewards.loc[:,:].plot(ax=ax)
                    # rewards.loc[:len(rewards)//2,:].plot(ax=ax)
                    # ax.set_xlabel("Cumulative reward")
                    # ax.set_ylabel("Frequency")
                    # ax.legend()
                    fig.close()
                else:
                    rewards = pd.DataFrame(data=np.array(rewards), columns=['ppo','gp'])
            
                    fig = plt.figure(figsize=set_size(width=1000.0))
                    ax = fig.add_subplot()
                    rewards['gp'].plot(ax=ax, kind='hist', alpha=0.7,color='tab:orange',bins=50)
                    rewards['ppo'].plot(ax=ax, kind='hist', color='tab:blue',bins=50)
                    ax.set_xlabel("Cumulative reward")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    means, stds = rewards.mean().values, rewards.std().values
                    ax.set_title('{} Obs \n Means: PPO {:.2f} GP {:.2f} \n Stds: PPO {:.2f} GP {:.2f} Exp {}'.format(len(rewards),*means,*stds, f))
                    fig.savefig(os.path.join(data_dir, "cumreward_hist_{}_{}.png".format(p['n_seeds'], f)), dpi=300)
                    plt.close()
                
                    fig = plt.figure(figsize=set_size(width=1000.0))
                    ax = fig.add_subplot()
                    sns.kdeplot(rewards['gp'].values, bw_method=0.2,ax=ax,color='tab:orange')
                    sns.kdeplot(rewards['ppo'].values, bw_method=0.2,ax=ax,color='tab:blue')
                    ax.set_xlabel("Cumulative reward")
                    ax.set_ylabel("KDE")
                    ax.legend()
                    means, stds = rewards.mean().values, rewards.std().values
                    ax.set_title('{} Obs \n Means: PPO {:.2f} GP {:.2f} \n Stds: PPO {:.2f} GP {:.2f} \n Exp {}'.format(len(rewards),*means,*stds, f))
                    ax.legend(labels=['gp','ppo'],loc=2)
                
                    KS, p_V = ks_2samp(rewards.values[:,0], rewards.values[:,1])
                    t, p_t = ttest_ind(rewards.values[:,0], rewards.values[:,1])
                    ks_text = AnchoredText("Ks Test: pvalue {:.2f} \n T Test: pvalue {:.2f}".format(p_V,p_t),loc=1,prop=dict(size=10))
                    ax.add_artist(ks_text)
                    fig.savefig(os.path.join(data_dir, "cumreward_density_{}_{}.png".format(p['n_seeds'],f)), dpi=300)
                    plt.close()


def runplot_holding(p):

    query = gin.query_parameter


    outputClass = p["outputClass"]
    tag = p["algo"]
    seed = p["seed"]
    if 'DQN' in tag:
        hp = p["hyperparams_model_dqn"]
        outputModels = p["outputModels_dqn"]
    elif 'PPO' in tag:
        hp = p["hyperparams_model_ppo"]
        outputModels = p["outputModels_ppo"]


    if hp is not None:
        outputModel = [exp.format(*hp) for exp in outputModels]
    else:
        outputModel = outputModels


    fig = plt.figure(figsize=set_size(width=1000.0, subplots=(2, 2)))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    axes = [ax1, ax2, ax3, ax4]

    # fig2 = plt.figure(figsize=set_size(width=1000.0, subplots=(2, 2)))
    # gs2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
    # ax12 = fig2.add_subplot(gs2[0])
    # ax22 = fig2.add_subplot(gs2[1])
    # ax32 = fig2.add_subplot(gs2[2])
    # ax42 = fig2.add_subplot(gs2[3])
    # axes2 = [ax12, ax22, ax32, ax42]
    

    fig3 = plt.figure(figsize=set_size(width=1000.0, subplots=(2, 2)))
    gs3 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig3)
    ax13 = fig3.add_subplot(gs3[0])
    ax23 = fig3.add_subplot(gs3[1])
    ax33 = fig3.add_subplot(gs3[2])
    ax43 = fig3.add_subplot(gs3[3])
    axes3 = [ax13, ax23, ax33, ax43]

    fig4 = plt.figure(figsize=set_size(width=1000.0, subplots=(2, 2)))
    gs4 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig4)
    ax14 = fig4.add_subplot(gs3[0])
    ax24 = fig4.add_subplot(gs3[1])
    ax34 = fig4.add_subplot(gs3[2])
    ax44 = fig4.add_subplot(gs3[3])
    axes4 = [ax14, ax24, ax34, ax44]

    for i, model in enumerate(outputModel):
        modelpath = "outputs/{}/{}".format(outputClass, model)
        # get the latest created folder "length"
        all_subdirs = [
            os.path.join(modelpath, d)
            for d in os.listdir(modelpath)
            if os.path.isdir(os.path.join(modelpath, d))
        ]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        length = os.path.split(latest_subdir)[-1]
        experiment = [
            exp
            for exp in os.listdir("outputs/{}/{}/{}".format(outputClass, model, length))
            if seed in exp
        ][0]
        data_dir = "outputs/{}/{}/{}/{}".format(outputClass, model, length, experiment)

        gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
        gin.bind_parameter('alpha_term_structure_sampler.generate_plot', p['generate_plot'])
        p['N_test'] = gin.query_parameter('%LEN_SERIES')
        
        gin.bind_parameter('Out_sample_vs_gp.rnd_state',p['random_state'])
        rng = np.random.RandomState(query("%SEED"))


        if query("%MV_RES"):
            action_space = ResActionSpace()
        else:
            action_space = ActionSpace()

        if gin.query_parameter('%MULTIASSET'):
            n_assets = len(gin.query_parameter('%HALFLIFE'))
            n_factors = len(gin.query_parameter('%HALFLIFE')[0])
            if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                # input_shape = (n_factors*n_assets+n_assets+1,1)
                input_shape = (int(n_factors*n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
            else:
                input_shape = (n_assets+n_assets+1,1)
                # input_shape = (int(n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
        else:
            if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                input_shape = (len(query('%F_PARAM')) + 1,)
            else:
                input_shape = (2,)


        if "DQN" in tag:
            train_agent = DQN(
                input_shape=input_shape, action_space=action_space, rng=rng
            )
            if p['n_dqn']:
                train_agent.model = load_DQNmodel(
                    data_dir, p['n_dqn'], model=train_agent.model
                )
            else:
                train_agent.model = load_DQNmodel(
                        data_dir, query("%N_TRAIN"), model=train_agent.model
                    )

        elif "PPO" in tag:
            
            train_agent = PPO(
                input_shape=input_shape, action_space=action_space, rng=rng
            )

            if p['ep_ppo']:
                train_agent.model = load_PPOmodel(data_dir, p['ep_ppo'], model=train_agent.model)
            else:
                train_agent.model = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"), model=train_agent.model)
        else:
            print("Choose proper algorithm.")
            sys.exit()

        if gin.query_parameter('%MULTIASSET'):
            if 'Short' in str(gin.query_parameter('%ENV_CLS')):
                env = ShortMultiAssetCashMarketEnv
            else:
                env = MultiAssetCashMarketEnv
        else:
            env = MarketEnv
        oos_test = Out_sample_vs_gp(
            savedpath=None,
            tag=tag[0],
            experiment_type=query("%EXPERIMENT_TYPE"),
            env_cls=env,
            MV_res=query("%MV_RES"),
            N_test=p['N_test']
        )

        res_df = oos_test.run_test(train_agent, return_output=True)


        if gin.query_parameter('%MULTIASSET'):

            plot_portfolio(res_df, tag[0], axes[i])
            # plot_action(res_df, tag[0], axes2[i])
            split = model.split("mv_res")
    
            axes[i].set_title(
                "_".join([split[-1]]).replace("_", " ") , fontsize=8
            )

            fig.suptitle('Holdings')
            
            if len(gin.query_parameter('%HALFLIFE'))==2:
    
                plot_2asset_holding(res_df, tag[0], axes3[i])
        
                axes3[i].set_title(
                    "_".join([split[-1]]).replace("_", " "), fontsize=8
                )
                fig3.suptitle('2 Asset Holdings')
                plt.close(fig4)
            else:
                plt.close(fig3)
                plt.close(fig4)

        else:

            plot_portfolio(res_df, tag[0], axes[i])
            # plot_action(res_df, tag[0], axes2[i])
            split = model.split("mv_res")
    
            axes[i].set_title(
                "_".join([split[-1]]).replace("_", " ") , fontsize=8
            )
            # axes2[i].set_title(
            #     "_".join(["mv_res", split[-1]]).replace("_", " "), fontsize=10
            # )
    
    
            plot_action(res_df, tag[0], axes3[i], hist=True)
    
            axes3[i].set_title(
                "_".join([split[-1]]).replace("_", " "), fontsize=8
            )
            
            plot_costs(res_df, tag[0], axes4[i], hist=False)
    
            axes4[i].set_title(
                "_".join([split[-1]]).replace("_", " "), fontsize=8
            )
    
    
            fig.suptitle('Holdings')
            # fig2.suptitle('Actions')
            fig3.suptitle('Res Actions')
            fig4.suptitle('Cumulative Costs')
    


def runplot_diagnostics(p):

    # tODO as of now this plot visualize diagnostics for each seed of the experiment set.
    outputClass = p["outputClass"]
    tag = p["algo"]
    if 'DQN' in tag:
        hp = p["hyperparams_model_dqn"]
        outputModels = p["outputModels_dqn"]
    elif 'PPO' in tag:
        hp = p["hyperparams_model_ppo"]
        outputModels = p["outputModels_ppo"]


    if hp is not None:
        outputModel = [exp.format(*hp) for exp in outputModels]
    else:
        outputModel = outputModels

    colors = ["blue", "y", "green", "black"]
    # colors = []
    random.seed(2212)  # 7156
    for _ in range(len(outputModel)):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        colors.append(color)

    for t in tag:

        
        var_plot_diagnostics = p['var_plot_diagnostics'] 

        for it, v in enumerate(var_plot_diagnostics):
            for k, out_mode in enumerate(outputModel):
                # read main folder
                fig = plt.figure(figsize=set_size(width=1000.0))
                # fig.subplots_adjust(wspace=0.2, hspace=0.6)
                ax = fig.add_subplot()

                modelpath = "outputs/{}/{}".format(outputClass, out_mode)

                # get the latest created folder "length"
                all_subdirs = [
                    os.path.join(modelpath, d)
                    for d in os.listdir(modelpath)
                    if os.path.isdir(os.path.join(modelpath, d))
                ]
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                length = os.path.split(latest_subdir)[-1]

                data_dir = "outputs/{}/{}/{}".format(outputClass, out_mode, length)

                # Recover and plot generated multi test OOS ----------------------------------------------------------------
                filtered_dir = [
                    dirname
                    for dirname in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(os.getcwd(), data_dir, dirname))
                ]
                logging.info(
                    "Plotting experiment {} for variable {}...".format(out_mode, v)
                )
                dfs = []
                for exp in filtered_dir:
                    exp_path = os.path.join(data_dir, exp)
                    array = np.load(os.path.join(exp_path, v))
                    df = pd.DataFrame(array, columns=[exp])

                    dfs.append(df)

                dataframe = pd.concat(dfs,axis=1)
                dataframe.plot(ax=ax)
                ax.set_title(out_mode)
                ax.set_ylabel(v.split('.npy')[0])
                ax.set_xlabel('Training time')
                

            logging.info("Plot saved successfully...")



if __name__ == "__main__":

    # Generate Logger-------------------------------------------------------------
    logger = generate_logger()

    # Read config ----------------------------------------------------------------
    p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))
    logging.info("Successfully read config file for Multi Test OOS...")

    if p["plot_type"] == "metrics":
        runplot_metrics(p)
    elif p["plot_type"] == "value":
        runplot_value(p)
    elif p["plot_type"] == "holding":
        runplot_holding(p)
    elif p["plot_type"] == "policy":
        runplot_policy(p)
    elif p["plot_type"] == "diagnostics":
        runplot_diagnostics(p)
    elif p["plot_type"] == "dist":
        runplot_distribution(p)
    elif p["plot_type"] == "distmulti":
        runmultiplot_distribution(p)
