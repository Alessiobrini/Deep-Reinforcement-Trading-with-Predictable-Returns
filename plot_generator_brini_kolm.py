# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:04:49 2021

@author: alessiobrini
"""

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
import re
import seaborn as sns
import gin
gin.enter_interactive_mode()
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

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
    plot_2asset_holding,
    plot_heatmap_holding
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


columnwidth=360
style = 'white' #darkgrid
params = {
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 11,
}
plt.rcParams.update(params)
sns.set_style(style)

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

            # read main folder
            fig = plt.figure(figsize=set_size(width=columnwidth)) 
            
            # fig.subplots_adjust(wspace=0.2, hspace=0.6)
            ax = fig.add_subplot()
            for k, out_mode in enumerate(outputModel):
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
                    
                    # ax.set_ylim(-1e09,1e09)
                    ax.set_ylim(1e06,4e06)


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
                    # ax.set_ylim(20, 150)
                    
                    
                # PERSONALIZE THE IMAGE WITH CORRECT LABELS
                ax.get_figure().gca().set_title("") # no title
                
                ax.set_xlabel('In-sample episodes')
                ax.set_ylabel('Reward')
                ax.legend(['model free','benchmark', 'residual'])
                
                fig.tight_layout()
                logging.info("Plot saved successfully...")
                
            fig.savefig("outputs/img_brini_kolm/exp_{}_{}.pdf".format(out_mode,v.split('_')[0]), dpi=300, bbox_inches="tight")


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


    fig = plt.figure(figsize=set_size(width=columnwidth))
    # ax1 = fig.add_subplot()
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    axes = [ax1, ax2]
    fig.subplots_adjust(hspace=0.25)
    
    model = outputModel[0]

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
        inputs = gin.query_parameter('%INPUTS')
        if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
            if 'sigma' in inputs and 'corr' in inputs:
                input_shape = (int(n_factors*n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
            else:
                input_shape = (int(n_factors*n_assets+n_assets+1),1)
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

    
    # DOUBLE PICTURE
    for i in range(2):
        ax = axes[i]
        oos_test.rnd_state = 1673
        # oos_test.rnd_state = np.random.choice(10000,1)
        # print(oos_test.rnd_state)
        res_df = oos_test.run_test(train_agent, return_output=True)
    
        if gin.query_parameter('%MULTIASSET'):
    
            plot_portfolio(res_df, tag[0], ax, tbox=False)
            if len(gin.query_parameter('%HALFLIFE'))>2:
                ax1.get_legend().remove()
            # ax.legend(['PPO','benchmark'], fontsize=8)
        else:
    
            plot_portfolio(res_df, tag[0], ax, tbox=False)
            ax.legend(['PPO','benchmark'], fontsize=8)
    
    ax.set_xlabel('Time')
    # ax.set_ylabel('Holding')
    fig.text(0.03, 0.5, 'Holding', ha='center', rotation='vertical')
    
    
    axes[0].get_xaxis().set_visible(False)
            
    # SINGLE PICTURE      
    # oos_test.rnd_state = 3524
    # oos_test.rnd_state = np.random.choice(10000,1)
    # print(oos_test.rnd_state)
    # res_df = oos_test.run_test(train_agent, return_output=True)

    # if gin.query_parameter('%MULTIASSET'):

    #     plot_portfolio(res_df, tag[0], ax1)
    #     split = model.split("mv_res")

    #     ax1.set_title(
    #         "_".join([split[-1]]).replace("_", " ") , fontsize=8
    #     )

    #     if len(gin.query_parameter('%HALFLIFE'))>2:
    #         ax1.get_legend().remove()
        
    #     fig.suptitle('Holdings')
    

    # else:

    #     plot_portfolio(res_df, tag[0], ax1, tbox=False)

    #     ax1.set_xlabel('Time')
    #     ax1.set_ylabel('Holding')
    #     ax1.legend(['PPO','benchmark'])

    # fig.tight_layout()
    if gin.query_parameter('%MULTIASSET'):
        # fig.savefig("outputs/img_brini_kolm/exp_{}_double_holding.pdf".format(model), dpi=300, bbox_inches="tight")
        pass
    else:
        fig.savefig("outputs/img_brini_kolm/exp_{}_single_holding.pdf".format(model), dpi=300, bbox_inches="tight")
    logging.info("Plot saved successfully...")

def runplot_multiholding(p):

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


    fig = plt.figure(figsize=set_size(width=columnwidth))
    # ax1 = fig.add_subplot()
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    axes = [ax1, ax2]
    fig.subplots_adjust(hspace=0.25)

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
            inputs = gin.query_parameter('%INPUTS')
            if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                if 'sigma' in inputs and 'corr' in inputs:
                    input_shape = (int(n_factors*n_assets+1+ (n_assets**2 - n_assets)/2+n_assets+1),1)
                else:
                    input_shape = (int(n_factors*n_assets+n_assets+1),1)
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
        oos_test.rnd_state = 1673
        res_df = oos_test.run_test(train_agent, return_output=True)

        plot_portfolio(res_df, tag[0], axes[i], tbox=False)
    
    axes[0].get_legend().remove()
    axes[0].get_xaxis().set_visible(False)
    axes[1].legend(['PPO 1','PPO 2','benchmark 1', 'benchmark 2'], fontsize=8, ncol=2)
    axes[1].set_xlabel('Time')
    fig.text(0.03, 0.5, 'Holding', ha='center', rotation='vertical')
    
    fig.savefig("outputs/img_brini_kolm/exp_{}_double_holding.pdf".format(model), dpi=300, bbox_inches="tight")
        

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
    if p['dist_to_plot'] == 'r':
        title = 'reward'
        rewards = Parallel(n_jobs=p['cores'])(delayed(parallel_test)(
                s, oos_test,train_agent,data_dir,p['fullpath']) for s in seeds)
    elif p['dist_to_plot'] == 'w':
        title= 'wealth'
        rewards = Parallel(n_jobs=p['cores'])(delayed(parallel_test_wealth)(
                s, oos_test,train_agent,data_dir,p['fullpath']) for s in seeds)

    if p['fullpath']:
        rewards_ppo = pd.concat(list(map(list, zip(*rewards)))[0],axis=1).cumsum()
        rewards_gp = pd.concat(list(map(list, zip(*rewards)))[1],axis=1).cumsum()
        fig = plt.figure(figsize=set_size(width=1000.0))
        ax = fig.add_subplot()
        rewards_ppo.mean(axis=1).plot(ax=ax, label='ppo_mean')
        rewards_gp.mean(axis=1).plot(ax=ax, label='gp_mean')
        ax.set_xlabel("Cumulative reward")
        ax.set_ylabel("Frequency")
        ax.legend()

        fig.close()
    else:
        rewards = pd.DataFrame(data=np.array(rewards), columns=['ppo','gp'])
        rewards.replace([np.inf, -np.inf], np.nan,inplace=True)

        cumdiff = rewards['ppo'].values - rewards['gp'].values
        # means, stds = rewards.mean().values, rewards.std().values
        # srs = means/stds
        KS, p_V = ks_2samp(rewards.values[:,0], rewards.values[:,1])
        t, p_t = ttest_ind(rewards.values[:,0], rewards.values[:,1])
        
        
        # DOUBLE PICTURES
        
        fig = plt.figure(figsize=set_size(width=columnwidth))
        # ax1 = fig.add_subplot()
        gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        fig.subplots_adjust(wspace=0.25)
        
        sns.kdeplot(rewards['ppo'].values, bw_method=0.2,ax=ax1,color='tab:blue')
        sns.kdeplot(rewards['gp'].values, bw_method=0.2,ax=ax1,color='tab:orange',linestyle="--")
        ax1.set_xlabel("Cumulative reward")
        ax1.set_ylabel("Kernel density")
        ax1.legend(labels=['PPO','benchmark']) 
        
        sns.kdeplot(cumdiff, bw_method=0.2,ax=ax2,color='tab:olive')
        ax2.set_xlabel("Cumulative diff reward")
        ax2.legend(labels=['difference']) 
        ax2.set_ylabel(None)
        # ks_text = AnchoredText("Ks Test: pvalue {:.2f} \n T Test: pvalue {:.2f}".format(p_V,p_t),loc=1,prop=dict(size=6))
        # ax2.add_artist(ks_text)
        
        fig.tight_layout()
        fig.savefig(os.path.join('outputs','img_brini_kolm', "kernel_densities_{}_{}.pdf".format(p['n_seeds'], outputModel)), dpi=300, bbox_inches="tight")
        
        # SINGLE PICTURES
        # fig = plt.figure(figsize=set_size(width=columnwidth))
        # ax = fig.add_subplot()
        # # sns.kdeplot(rewards['gp'].values, bw_method=0.2,ax=ax,color='tab:orange')
        # sns.kdeplot(cumdiff, bw_method=0.2,ax=ax,color='tab:blue')
        # ax.set_xlabel("Cumulative reward diff")
        # ax.set_ylabel("KDE")
        # ax.set_title('{} Obs \n Means: PPO {:.2f} GP {:.2f} \n Stds: PPO {:.2f} GP {:.2f} \n SR:  PPO {:.2f} GP {:.2f} \n Exp {}'.format(len(rewards),*means,*stds, *srs, outputModel))
        # ax.legend(labels=['gp','ppo'],loc=2)
        # ks_text = AnchoredText("Ks Test: pvalue {:.2f} \n T Test: pvalue {:.2f}".format(p_V,p_t),loc=1,prop=dict(size=10))
        # ax.add_artist(ks_text)
        # fig.savefig(os.path.join('outputs','img_brini_kolm', "cumreward_diff_density_{}_{}.pdf".format(p['n_seeds'], outputModel)), dpi=300)
        # plt.close()
            
    
        # fig = plt.figure(figsize=set_size(width=columnwidth))
        # ax = fig.add_subplot()
        # sns.kdeplot(rewards['gp'].values, bw_method=0.2,ax=ax,color='tab:orange',linestyle="--")
        # sns.kdeplot(rewards['ppo'].values, bw_method=0.2,ax=ax,color='tab:blue')
        # ax.set_xlabel("Cumulative reward")
        # ax.set_ylabel("KDE")
        # ax.set_title('{} Obs \n Means: PPO {:.2f} GP {:.2f} \n Stds: PPO {:.2f} GP {:.2f} \n SR:  PPO {:.2f} GP {:.2f} \n Exp {}'.format(len(rewards),*means,*stds, *srs, outputModel))
        # ax.legend(labels=['gp','ppo'],loc=2)        
        # ks_text = AnchoredText("Ks Test: pvalue {:.2f} \n T Test: pvalue {:.2f}".format(p_V,p_t),loc=1,prop=dict(size=10))
        # ax.add_artist(ks_text)
        # fig.savefig(os.path.join('outputs','img_brini_kolm', "cumreward_density_{}_{}.pdf".format(p['n_seeds'], outputModel)), dpi=300)
        # plt.close()


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


def runplot_time(p):


    modeltag = p['modeltag']
    outputClass = p["outputClass"]
    
    if isinstance(modeltag,list):
        assets = []
        runtimes = []
        for mtag in modeltag:
            folders = [p for p in glob.glob(os.path.join('outputs',outputClass,'{}'.format(mtag.format('*'))))]
            length = os.listdir(folders[0])[0]


            n_assets= [int(f.split('n_assets_')[-1].split('_')[0]) for f in folders]
            times = []    

            for main_f in folders:    
                length = os.listdir(main_f)[0]
                for f in os.listdir(os.path.join(main_f,length)):
                    if 'seed_{}'.format(p['seed']) in f:
                        data_dir = os.path.join(main_f,length,f)
                        
                        file = open(os.path.join(data_dir,'runtime.txt'),'r')
                        txt = file.read()
                        times.append(float(re.findall("\d+\.\d+", txt)[0]))


            ltup = list(zip(n_assets,times))
            ltup.sort(key=lambda y: y[0])
            a,t = zip(*ltup)
            
            
            assets.append(a)
            runtimes.append(t)
        
    
        fig = plt.figure(figsize=set_size(width=columnwidth))
        ax = fig.add_subplot()
        
        for i in range(len(assets)):
            
            ax.plot(assets[i],runtimes[i])
        
        # ax.plot(assets[i],assets[i])
        
        ax.set_ylabel('Runtime for 100 episodes (minutes)')
        ax.set_xlabel('Number of assets')

        ax.legend([('-').join(mt.split('_')[-2:]) for mt in modeltag])
        

        # # objective function
        # def objective(x, a, b, c):
        # 	return a * x + b
        # popt, _ = curve_fit(objective, assets[i],runtimes[i])
        # corr, _ = pearsonr(assets[i],runtimes[i])

        fig.tight_layout()
        fig.savefig(os.path.join('outputs','img_brini_kolm', "multiruntime_{}.pdf".format(modeltag[0].split('_{}_')[0])), dpi=300, bbox_inches="tight")
        

    else:
        # folders = [p for p in glob.glob(os.path.join('outputs',outputClass,'{}*'.format(modeltag)))]
        folders = [p for p in glob.glob(os.path.join('outputs',outputClass,'{}'.format(modeltag.format('*'))))]
        length = os.listdir(folders[0])[0]
        pdb.set_trace()
        
        
        n_assets= [int(f.split('n_assets_')[-1].split('_')[0]) for f in folders]
        times = []    
        # pdb.set_trace()
        for main_f in folders:    
            length = os.listdir(main_f)[0]
            for f in os.listdir(os.path.join(main_f,length)):
                if 'seed_{}'.format(p['seed']) in f:
                    data_dir = os.path.join(main_f,length,f)
                    
                    file = open(os.path.join(data_dir,'runtime.txt'),'r')
                    txt = file.read()
                    times.append(float(re.findall("\d+\.\d+", txt)[0]))
                    
        # pdb.set_trace()
        ltup = list(zip(n_assets,times))
        ltup.sort(key=lambda y: y[0])
        n_assets,times = zip(*ltup)
    
    
        fig = plt.figure(figsize=set_size(width=columnwidth))
        ax = fig.add_subplot()
        
        ax.plot(n_assets,times)
        
        ax.set_ylabel('Runtime for 100 episodes (minutes)')
        ax.set_xlabel('Number of assets')
    
        fig.tight_layout()
        fig.savefig(os.path.join('outputs','img_brini_kolm', "runtime_{}_{}.pdf".format(p['n_seeds'], modeltag)), dpi=300, bbox_inches="tight")
        
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

    fig = plt.figure(figsize=set_size(width=columnwidth))
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
        # ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join('outputs','img_brini_kolm', "ppo_policy_{}_{}.pdf".format(p['seed'], outputModel)), dpi=300, bbox_inches="tight")
        
    else:
        print("Choose proper algorithm.")
        sys.exit()



if __name__ == "__main__":

    # Generate Logger-------------------------------------------------------------
    logger = generate_logger()

    # Read config ----------------------------------------------------------------
    p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))
    logging.info("Successfully read config file for Multi Test OOS...")

    if p["plot_type"] == "metrics":
        runplot_metrics(p)
    # elif p["plot_type"] == "value":
    #     runplot_value(p)
    elif p["plot_type"] == "holding":
        runplot_holding(p)
    elif p["plot_type"] == "multiholding":
        runplot_multiholding(p)
    elif p["plot_type"] == "policy":
        runplot_policy(p)
    # elif p["plot_type"] == "diagnostics":
    #     runplot_diagnostics(p)
    elif p["plot_type"] == "dist":
        runplot_distribution(p)
    # elif p["plot_type"] == "distmulti":
    #     runmultiplot_distribution(p)
    elif p['plot_type'] == 'runtime':
        runplot_time(p)
