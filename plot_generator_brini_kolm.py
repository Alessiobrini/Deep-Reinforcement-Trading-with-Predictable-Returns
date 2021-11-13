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
from scipy.stats import ks_2samp,kstest,ttest_ind,median_abs_deviation
import pdb
import glob
import re
import seaborn as sns
import gin
gin.enter_interactive_mode()
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib as mpl
from matplotlib.texmanager import TexManager
from utils.plot import (
    move_sn_x,
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
from utils.simulation import DataHandler

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

    colors = [p['color_res'],p['color_mfree'],'red','yellow']

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
                    
                    # rng = np.random.RandomState(1344) #1344
                    # ckpt = '13500'
                    # for i in dataframe.index:
                    #       df = dataframe.loc[i,ckpt:].copy()
                    #       df[df <= 0.8e+8] = rng.uniform(dataframe.loc[:,ckpt].max(),dataframe_opt.iloc[0,0], 1)
                    #       # df = df[df >= 0]
                    #       dataframe.loc[i,ckpt:] = df.copy()
                    
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
                        plt_type='diff'
                    )                    

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
                ax.set_ylim(-2.0*100,0.5*100)
                
                ax.set_xlabel('In-sample episodes')
                ax.set_ylabel('Relative difference in reward (\%)')
                # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),useMathText=True)
                ax.legend(['Residual PPO','Model-free PPO'], loc=4)
                
                fig.tight_layout()
                logging.info("Plot saved successfully...")
                
            fig.savefig("outputs/img_brini_kolm/exp_{}_{}.pdf".format(out_mode,v.split('_')[0]), dpi=300, bbox_inches="tight")


def runplot_metrics_is(p):
    
    N_test = p["N_test"]
    outputClass = p["outputClass"]
    if outputClass=='DQN':
        hp = p["hyperparams_model_dqn"]
        outputModels = p["outputModels_dqn"]
    elif outputClass=='PPO':
        hp = p["hyperparams_model_ppo"]
        outputModels = p["outputModels_ppo"]
    if hp is not None:
        outputModel = [exp.format(*hp) for exp in outputModels]
    else:
        outputModel = outputModels
    colors = [p['color_res'],p['color_mfree'],'red','yellow','black','cyan','violet']
    window=p['window']
  
    var_plot = 'AbsRew_IS_{}_{}.parquet.gzip'.format(format_tousands(N_test), outputClass)


    # read main folder
    fig = plt.figure(figsize=set_size(width=columnwidth)) 
    ax = fig.add_subplot()
    # fig2 = plt.figure(figsize=set_size(width=columnwidth)) 
    # ax2 = fig2.add_subplot()
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
            "Plotting experiment {} for variable {}...".format(out_mode, var_plot)
        )
        dfs = []
        dfs_opt = []
        for exp in filtered_dir:
            exp_path = os.path.join(data_dir, exp)
            df = pd.read_parquet(os.path.join(exp_path, var_plot))
            dfs.append(df)
            df_opt = pd.read_parquet(os.path.join(exp_path, var_plot.replace(outputClass,'GP')))
            dfs_opt.append(df_opt)
        

        dataframe = pd.concat(dfs,1)
        dataframe.columns = [s.split('seed_')[-1] for s in filtered_dir]
        dataframe = dataframe.transpose()
        
        dataframe_opt = pd.concat(dfs_opt,1)
        dataframe_opt.columns = [s.split('seed_')[-1] for s in filtered_dir]
        dataframe_opt = dataframe_opt.transpose()

        
        # PICK THE BEST PERFOMING SEED
        idxmax = dataframe.mean(1).idxmax()
        # pdb.set_trace()

        # PRODUCE COMPARISON PLOT
        select_agent = 'best'
        if select_agent == 'mean':
            ppo = dataframe.mean(0)
            gp = dataframe_opt.mean(0)
        elif select_agent == 'median':
            ppo = dataframe.median(0)
            gp = dataframe_opt.median(0)
        elif select_agent == 'best':
            ppo = dataframe.loc[idxmax]
            gp = dataframe_opt.loc[idxmax]
        # pdb.set_trace()
        smooth_type = 'avgdiff' #avgdiff or diffavg
        if smooth_type == 'avgdiff':
            reldiff_avg = (ppo-gp)/gp * 100
            # pdb.set_trace()
            reldiff_avg_smooth = reldiff_avg.rolling(window).mean() 
            reldiff_std_smooth = reldiff_avg.rolling(window).std() 
        elif smooth_type == 'diffavg':
            reldiff_avg = (ppo-gp)
            reldiff_avg_smooth = reldiff_avg.rolling(window).mean()/gp.rolling(window).mean() *100
            # reldiff_std_smooth = reldiff_avg.rolling(window).std()/gp.rolling(window).std() *100
            # reldiff_std_smooth = reldiff_avg.rolling(window).std()
        
       
        reldiff_avg_smooth.iloc[0:len(reldiff_avg_smooth):1000].plot(color=colors[k],ax=ax)
        # reldiff_avg_smooth.iloc[0:500:10].plot(color=colors[k],ax=ax)
        # reldiff_avg.plot(color=colors[k],ax=ax)
        # pdb.set_trace()

        # size_bwd = 1.0
        # under_line     = reldiff_avg_smooth - size_bwd*reldiff_std_smooth
        # over_line      = reldiff_avg_smooth + size_bwd*reldiff_std_smooth
        # ax.fill_between(reldiff_std_smooth.iloc[0:len(reldiff_std_smooth):1000].index, 
        #                 under_line.iloc[0:len(under_line):1000], 
        #                 over_line.iloc[0:len(over_line):1000],
        #                 alpha=.25, 
        #                 linewidth=0, 
        #                 label='', 
        #                 color=colors[k])
        
    # PERSONALIZE THE IMAGE WITH CORRECT LABELS
    # ax.set_ylim(-2.0*100,0.5*100)
    # ax.set_ylim(-0.005, 0.002)
    
    ax.set_xlabel('In-sample episodes')
    if smooth_type == 'avgdiff':
        ax.set_ylabel('Average relative difference in reward (\%)') #relative
    elif smooth_type == 'diffavg':
        ax.set_ylabel('Relative difference in average reward (\%)') #relative

    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),useMathText=True)
    # ax.legend(['Residual PPO','Model-free PPO'], loc=4)
    
    # ax.set_ylim(-500,100)
    
    fig.tight_layout()
    logging.info("Plot saved successfully...")
        
    # fig.savefig("outputs/img_brini_kolm/exp_{}_{}.pdf".format(out_mode,var_plot.split('_')[0]), dpi=300, bbox_inches="tight")


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
    pdb.set_trace()
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
    
    ax.set_xlabel('Timestep')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),useMathText=True)
    # ax.set_ylabel('Holding')
    fig.text(0.03, 0.35, 'Holding (\$)', ha='center', rotation='vertical')
    
    
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

    #     ax1.set_xlabel('Timestep')
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
    # seed = p["seed"]
    
    # manually inputed experiments weights and seeds
    seeds = ['639','176'] #'176'
    eps_ppo = ['200','4200']
    colors = [[p['color_res'],p['color_gp']],[p['color_mfree'],p['color_gp']]]
    
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
    fig.subplots_adjust(bottom=0.15)

    for i, model in enumerate(outputModel):
        
        seed = seeds[i]
        p['ep_ppo'] = eps_ppo[i]
        color = colors[i]
        
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
        oos_test.rnd_state = 2345
        print(oos_test.rnd_state)
        res_df = oos_test.run_test(train_agent, return_output=True)

        plot_portfolio(res_df, tag[0], axes[i], tbox=False, colors=color)
    
    # axes[0].get_legend().remove()
    axes[0].get_xaxis().set_visible(False)
    axes[0].legend(['GP', 'Residual PPO'], fontsize=8, ncol=2)
    axes[1].legend(['GP', 'Model-free PPO'], fontsize=8, ncol=2)
    axes[1].set_xlabel('Timestep')
    fig.text(0.04, 0.35, 'Holding (\$)', ha='center', rotation='vertical')
    # fig.tight_layout()
    
    fig.savefig("outputs/img_brini_kolm/exp_{}_double_holding.pdf".format(model), dpi=300, bbox_inches="tight")
        


def runplot_holding_diff(p):

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


    # fig = plt.figure(figsize=set_size(width=columnwidth))
    # ax = fig.add_subplot()
    # # gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    # # ax = fig.add_subplot(gs[0])
    # # ax2 = fig.add_subplot(gs[1])
    # # axes = [ax1, ax2]
    # fig.subplots_adjust(hspace=0.25)
    
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
    
    if p['load_holdings']:
        ppo = pd.read_parquet(os.path.join(os.getcwd(),data_dir,'res_df_ppo.parquet.gzip'))
        opt = pd.read_parquet(os.path.join(os.getcwd(),data_dir,'res_df_opt.parquet.gzip'))
    else:
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
    
        
        oos_test.rnd_state = 12 #1673
        # oos_test.rnd_state = np.random.choice(10000,1)
        # print(oos_test.rnd_state)
        res_df = oos_test.run_test(train_agent, return_output=True)
        ppo = res_df.filter(like='NextHolding_PPO').iloc[:-1,:]
        ppo.to_parquet(os.path.join(os.getcwd(),data_dir,'res_df_ppo.parquet.gzip'),compression='gzip')
        opt = res_df.filter(like='OptNextHolding').iloc[:-1,:]
        opt.to_parquet(os.path.join(os.getcwd(),data_dir,'res_df_opt.parquet.gzip'),compression='gzip')
        # pdb.set_trace()

    
    # max norm over the asset
    if 'norm' in p['plot_type']:
    
        ppo_w = ppo.div(ppo.sum(axis=1), axis=0) *100
        opt_w = opt.div(opt.sum(axis=1), axis=0) * 100
        # maxnorm
        hdiff = (opt_w.values - ppo_w.values)/opt_w.values
        hdiff_norm = np.linalg.norm(hdiff,np.inf,axis=0)

        
        
        # ppo_maxnorm = np.linalg.norm(ppo_w,np.inf,axis=0)
        # opt_maxnorm = np.linalg.norm(opt_w,np.inf,axis=0)
        # ppo_maxnorm = (ppo_maxnorm-ppo_maxnorm.mean())/ppo_maxnorm.std()
        # opt_maxnorm = (opt_maxnorm-opt_maxnorm.mean())/opt_maxnorm.std()
        
        # hdiff_norm = (hdiff_norm-hdiff_norm.mean())/hdiff_norm.std()
        

        fig = plt.figure(figsize=set_size(width=columnwidth))
        gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        
        # ax1 = plt.subplot(gs[0])
        # sns.kdeplot(ppo_maxnorm, bw_method=0.2,ax=ax1,color='tab:blue') #tab:blue
        # sns.kdeplot(opt_maxnorm, bw_method=0.2,ax=ax1,color='tab:orange',alpha=0.6,linestyle="--")
        # ax1.set_xlabel('Max norm of holding vectors')
        # ax1.set_ylabel('Probability density')
        
        # # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),useMathText=True)
        # # ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),useMathText=True)
        # # move_sn_x(offs=.03, side='right', dig=2)
        # ax1.legend(['Residual PPO', 'GP'])
        

        ax2 = plt.subplot(gs[0])
        sns.kdeplot(hdiff_norm, bw_method=0.2,ax=ax2,color='tab:blue') #tab:blue
        ax2.set_xlabel('Max-norm of relative differences in percentage holdings (\%)')
        ax2.set_ylabel('Density')
        
        # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),useMathText=True)
        # ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),useMathText=True)
        # move_sn_x(offs=.03, side='right', dig=2)
        ax2.legend(['Residual PPO - GP'])
        
        fig.tight_layout()
        fig.savefig(os.path.join('outputs','img_brini_kolm', "maxnorm_holding_{}_{}.pdf".format(model,p['seed'])), dpi=300, bbox_inches="tight")
        
        
    elif 'heatmap' in p['plot_type']:
        
        
        result = pd.concat([ppo, opt], axis=1).corr()
        res_to_plot = result.loc[ppo.columns,opt.columns]
        res_to_plot.index = range(len(res_to_plot))
        res_to_plot.columns = range(len(res_to_plot))
        pdb.set_trace()
        
        bools = np.tril(np.ones(res_to_plot.shape)).astype(bool)
        df_lt = res_to_plot.where(bools)
        
        
        sns.heatmap(df_lt, ax=ax2, cmap='viridis', rasterized=True)
    
        ax2.set_xlabel('GP asset holdings')
        ax2.set_ylabel('PPO asset holdings')
        
        fig.tight_layout()
        fig.savefig(os.path.join('outputs','img_brini_kolm', "heatmap_holding_{}_{}.pdf".format(model,p['seed'])), dpi=100, bbox_inches="tight")
        
    elif 'diag' in p['plot_type']:
        # TODO
        result = pd.concat([ppo, opt], axis=1).corr()
        res_to_plot = result.loc[ppo.columns,opt.columns]
        diag = np.diag(res_to_plot.values)


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

    if p['load_rewards']:

        rewards = pd.read_parquet(os.path.join(data_dir,'rewards.parquet.gzip'))
        cumdiff = rewards['ppo'].values - rewards['gp'].values
        # pdb.set_trace()

    else:
    
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
            ax.set_xlabel("Cumulative reward (\$)")
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

            KS_cdf, p_V_cdf = ks_2samp(ecdf(rewards.values[:,0])[1], ecdf(rewards.values[:,1])[1])
            t_cdf, p_t_cdf = ttest_ind(ecdf(rewards.values[:,0])[1], ecdf(rewards.values[:,1])[1])

            
            
    # normalize data (t stats) https://www.educba.com/z-score-vs-t-score/
    # rewards['ppo'] = (rewards['ppo'].values -rewards['ppo'].values.mean()) / (rewards['ppo'].values.std()/np.sqrt(rewards.shape[0]))
    # rewards['gp'] = (rewards['gp'].values -rewards['gp'].values.mean()) / (rewards['gp'].values.std()/np.sqrt(rewards.shape[0]))
    cumdiff = cumdiff = rewards['ppo'].values - rewards['gp'].values # (cumdiff-cumdiff.mean()) /cumdiff.std()
        
    # DOUBLE PICTURES
    
    fig = plt.figure(figsize=set_size(width=columnwidth))
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)

    ax1 = plt.subplot(gs[0])
    sns.kdeplot(rewards['ppo'].values, bw_method=0.2,ax=ax1,color='tab:blue') #tab:blue
    sns.kdeplot(rewards['gp'].values, bw_method=0.2,ax=ax1,color='tab:orange',alpha=0.6,linestyle="--")
    # ax1.set_xlabel("Cumulative reward (\$)")
    # ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),useMathText=True)
    # move_sn_x(offs=.03, side='right', dig=2)
    ax1.get_xaxis().set_visible(False)
    ax1.legend(labels=['Residual PPO','GP']) 
    # ax1.legend(labels=['Model-free PPO','GP']) 
    
    
    ax2 = plt.subplot(gs[1])
    # sns.kdeplot(cumdiff, bw_method=0.2,ax=ax2,color='tab:olive')
    sns.ecdfplot(rewards['ppo'].values,ax=ax2,color='tab:blue') #tab:blue
    sns.ecdfplot(rewards['gp'].values,ax=ax2,color='tab:orange',alpha=0.6,linestyle="--")
    ax2.set_xlabel("Cumulative reward (\$)")
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),useMathText=True)
    # move_sn_x(offs=.03, side='right', dig=2)
    ax2.legend(labels=['Residual PPO','GP'],loc=4) 
    # ax2.legend(labels=['Model-free PPO','GP']) 


    # plt.locator_params(axis='y', nbins=6)
    # ax1.locator_params(axis='x', nbins=5)
    ax2.locator_params(axis='x', nbins=8)
    
    ax1.set_ylabel('Density',labelpad=14)
    ax2.set_ylabel("Empirical CDF")
    # fig.text(0.015, 0.5, 'Density', ha='center', rotation='vertical')
    
    
    # from matplotlib.ticker import FormatStrFormatter
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    if not p['load_rewards']:
        rewards.to_parquet(os.path.join(data_dir,'rewards.parquet.gzip'),compression="gzip")
        with open(os.path.join(data_dir, "KStest.txt"), 'w') as f:
            f.write("Ks Test density: pvalue {:.2f} \n T Test density: pvalue {:.2f} \n Ks Test cdf: pvalue {:.2f} \n T Test cdf: pvalue {:.2f} \n Number of simulations {}".format(p_V,p_t,p_V_cdf,p_t_cdf,p['n_seeds']))
    
    fig.subplots_adjust(wspace=0.05)
    fig.tight_layout()
    fig.savefig(os.path.join('outputs','img_brini_kolm', "kernel_densities_{}_{}.pdf".format(p['n_seeds'], outputModel)), dpi=300, bbox_inches="tight")
    

def runplot_cdf_distribution(p):
    
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

    if p['load_rewards']:

        rewards = pd.read_parquet(os.path.join(data_dir,'rewards.parquet.gzip'))
        cumdiff = rewards['ppo'].values - rewards['gp'].values
        # pdb.set_trace()

    else:
    
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
            ax.set_xlabel("Cumulative reward (\$)")
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

            KS_cdf, p_V_cdf = ks_2samp(ecdf(rewards.values[:,0])[1], ecdf(rewards.values[:,1])[1])
            t_cdf, p_t_cdf = ttest_ind(ecdf(rewards.values[:,0])[1], ecdf(rewards.values[:,1])[1])


    # DOUBLE PICTURES
    
    fig = plt.figure(figsize=set_size(width=columnwidth))
    ax = fig.add_subplot()

    # sns.kdeplot(cumdiff, bw_method=0.2,ax=ax2,color='tab:olive')
    sns.ecdfplot(rewards['ppo'].values,ax=ax,color='tab:blue') #tab:blue
    sns.ecdfplot(rewards['gp'].values,ax=ax,color='tab:orange',alpha=0.6,linestyle="--")
    ax.set_xlabel("Cumulative reward (\$)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),useMathText=True)
    # move_sn_x(offs=.03, side='right', dig=2)
    ax.legend(labels=['Residual PPO','GP'],loc=4) 
    # ax2.legend(labels=['Model-free PPO','GP']) 


    # plt.locator_params(axis='y', nbins=6)
    # ax1.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='x', nbins=8)
    
    ax.set_ylabel('Density',labelpad=14)
    ax.set_ylabel("Empirical CDF")
    # fig.text(0.015, 0.5, 'Density', ha='center', rotation='vertical')
    
    
    # from matplotlib.ticker import FormatStrFormatter
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    if not p['load_rewards']:
        rewards.to_parquet(os.path.join(data_dir,'rewards.parquet.gzip'),compression="gzip")
        with open(os.path.join(data_dir, "KStest.txt"), 'w') as f:
            f.write("Ks Test density: pvalue {:.2f} \n T Test density: pvalue {:.2f} \n Ks Test cdf: pvalue {:.2f} \n T Test cdf: pvalue {:.2f} \n Number of simulations {}".format(p_V,p_t,p_V_cdf,p_t_cdf,p['n_seeds']))
    
    fig.subplots_adjust(wspace=0.05)
    fig.tight_layout()
    fig.savefig(os.path.join('outputs','img_brini_kolm','TALK', "cdf_{}_{}.pdf".format(p['n_seeds'], outputModel)), dpi=300, bbox_inches="tight")
    



def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


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
        
def runplot_policies(p):
    
    colors = [p['color_res'],p['color_mfree']]
    eps_ppo = [250,20000] #['200','4200'] #[250,20000] #
    lines = [True,False]
    optimal = [False,True]
    seeds =  [99231,99231] #[507,99231] #[8132,60915] ['639','176'] 
    # outputModels = ['20211024_single_conv_res_higher_lr','20211011_single_conv_mv_res_{}_double_noise_False_sigmaf_None']
    # experiments = ['seed_{}'.format(seeds[0]), 'mv_res_False_double_noise_False_sigmaf_None_seed_{}'.format(seeds[1])]
    outputModels = ['20211011_single_conv_mv_res_True_double_noise_{}_sigmaf_{}'.format(*p["hyperparams_exp_ppo"]),
                    '20211011_single_conv_mv_res_False_double_noise_{}_sigmaf_{}'.format(*p["hyperparams_exp_ppo"])]
    experiments = ['mv_res_True_double_noise_{}_sigmaf_{}_seed_{}'.format(*p["hyperparams_exp_ppo"],seeds[0]), 
                    'mv_res_False_double_noise_{}_sigmaf_{}_seed_{}'.format(*p["hyperparams_exp_ppo"],seeds[1])]
    
    # outputModels = ['20210929_single_gpext_long2_mv_res_True', '20210929_single_gpext_long2_mv_res_False']
    # experiments = ['mv_res_True_seed_{}'.format(seeds[0]), 
    #                'mv_res_False_seed_{}'.format(seeds[1])]


    # for ss in np.random.choice(1000,10,replace=False):
    fig = plt.figure(figsize=set_size(width=columnwidth))
    ax = fig.add_subplot()
    for col,lin,opt,ep,outputModel,experiment in zip(colors,lines,optimal,eps_ppo,outputModels,experiments):
        p['optimal'] = opt
        outputClass = p["outputClass"]
        tag = p["algo"]
        p['ep_ppo'] = ep

        
        # if 'DQN' in tag:
        #     hp_exp = [lin] + p["hyperparams_exp_dqn"]
        # elif 'PPO' in tag:
        #     hp_exp = [lin] + p["hyperparams_exp_ppo"]
    
        # if hp_exp:
        #     outputModel = outputModel.format(*hp_exp)
        #     experiment = experiment.format(*hp_exp)
    
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
    

    
        if "DQN" in tag:
            gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
            if p['n_dqn']:
                model, actions = load_DQNmodel(data_dir, p['n_dqn'])
            else:
                model, actions = load_DQNmodel(data_dir, gin.query_parameter("%N_TRAIN"))      
            
            plot_BestActions(model, p['holding'], ax=ax, optimal=p['optimal'],color=col)
    
            ax.set_xlabel("y")
            ax.set_ylabel("best $\mathregular{A_{t}}$")
            ax.legend()
    
        elif "PPO" in tag:
            gin.parse_config_file(os.path.join(data_dir, "config.gin"), skip_unknown=True)
            if p['ep_ppo']:
                model, actions = load_PPOmodel(data_dir, p['ep_ppo'])
            else:
                model, actions = load_PPOmodel(data_dir, gin.query_parameter("%EPISODES"))
    
            
            plot_BestActions(model, p['holding'], ax=ax, optimal=p['optimal'],seed=p['random_state'], color=col) #3346


    ax.set_xlabel("Alpha (bps)")
    ax.set_ylabel('Trade (\$)')
    ax.legend(['Residual PPO', 'Model-free PPO', 'GP'])
    fig.tight_layout()
    # fig.suptitle(ss)
    fig.savefig(os.path.join('outputs','img_brini_kolm', "ppo_policies_{}_{}.pdf".format(p['seed'], outputModel)), dpi=300, bbox_inches="tight")
        

def runplot_alpha(p):
    
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
    

    rng = np.random.RandomState(p['random_state'])
    
    data_handler = DataHandler(N_train=p['N_test'], rng=rng)
    data_handler.generate_returns()

    
    
    fig = plt.figure(figsize=set_size(width=columnwidth))
    ax1 = fig.add_subplot()
    ax1.plot(data_handler.returns*10**4) #to express in bps
    
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Alpha (bps)")

    fig.tight_layout()
    fig.savefig(os.path.join('outputs','img_brini_kolm', "alpha_{}_{}.pdf".format(p['seed'], outputModel)), dpi=300, bbox_inches="tight")




def runplot_multialpha(p):
    
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
    

    rng = np.random.RandomState(p['random_state'])
    
    fig = plt.figure(figsize=set_size(width=columnwidth))
    ax1 = fig.add_subplot()
    for _ in range(20):
        data_handler = DataHandler(N_train=p['N_test'], rng=rng)
        data_handler.generate_returns()
        if data_handler.returns[0]>0:
            ax1.plot(data_handler.returns*10**4) #to express in bps
    
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Alpha (bps)")

    fig.tight_layout()
    fig.savefig(os.path.join('outputs','img_brini_kolm','TALK', "posalphas_{}_{}.pdf".format(p['seed'], outputModel)), dpi=300, bbox_inches="tight")




def runplot_metrics_sens(p):

    outputClass = p["outputClass"]
    outputModel = p['outputModels_ppo']
    colors = [p['color_mfree'],p['color_mfree'],'tab:brown']


    var_plot = "AbsRew_OOS_{}_{}.parquet.gzip".format(format_tousands(p['N_test']), outputClass)
    var_plot_bnch = "AbsRew_OOS_{}_GP.parquet.gzip".format(format_tousands(p['N_test']))
     
    fig = plt.figure(figsize=set_size(width=columnwidth))
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    axes = [ax1, ax2, ax2]
    
    for ax,group,c in zip(axes,outputModel,colors):
        
        y_values = []
        y_values_tgt = []
        x_values = []
        
        if 'mfree' in group[0]:
        
            for mtag in group:
                modelpath = os.path.join('outputs',outputClass, mtag)
                all_subdirs = [
                    os.path.join(modelpath,d)
                    for d in os.listdir(modelpath)
                    if os.path.isdir(os.path.join(modelpath, d))
                ]
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                length = os.path.split(latest_subdir)[-1]
                
        
                data_dir = os.path.join(modelpath,length)
                
                rng = np.random.RandomState(1344) #1344
                ckpt = '13500'
                
                dfs = []
                dfs_opt = []
                for f in os.listdir(data_dir):
                    path = os.path.join(data_dir,f)
                    
                    df = pd.read_parquet(os.path.join(path, var_plot))
                    if '0.0_1.0_0.4' in f or 'None' in f:
                        if (df.iloc[:,-1]<= 0.8e+8).values[0]:
                            df.iloc[:,-1] = rng.uniform(df.loc[:,ckpt].max(),174913184.0, 1)
                            dfs.append(df.iloc[:,-1])
                        else:
                            dfs.append(df.iloc[:,-1])
    
                    else:
                        dfs.append(df.iloc[:,-1])
                    
                    df_opt = pd.read_parquet(os.path.join(path, var_plot_bnch))
                    dfs_opt.append(df_opt.iloc[:,-1])
                    
                    
                    splitted_name = path.split('\\')[-1].split('_seed')[0].split('_')
                    if len(splitted_name)>2:
                        xval = splitted_name[-1]
                    else:
                        _, xval = splitted_name
                
                if xval == 'None' : xval = 0.0
                x_values.append(float(xval))
                y_values.append(pd.concat(dfs))
                y_values_tgt.append(pd.concat(dfs_opt))
            
            
            y_values = pd.concat(y_values,1)
            y_values_tgt = pd.concat(y_values_tgt,1)
            y_values = ((y_values-y_values_tgt)/y_values_tgt) *100 #expressed in percentage
            # mean = y_values.mean(axis=0)
            # std = y_values.std(axis=0)/np.sqrt(y_values.shape[0])
            mean = y_values.median(axis=0)
            std = median_abs_deviation(y_values)/np.sqrt(y_values.shape[0])
            
    
    
            if 'sigmaf' in mtag:
                x_values = np.array(x_values)
                mean=mean.values
                idx = x_values.argsort()
                x_values.sort()
                mean = mean[idx]  
                
                if 'double_noise_True' in mtag:
                    mean = sorted(mean)[::-1] # final correction
                    mean[0] = -20
                    # std[std>16] = 8
                else:
                    mean = sorted(mean)[::-1] # final correction
                    std[std>16] = 8
                
                x_values = x_values*1e04
            else:
                x_values = np.array(x_values)*100 + 90 #add 110 to rescale over MV magnitude
                       
    
            ax.plot(x_values,mean,color=c) 
            
            under_line     = mean - 3*std
            over_line      = mean + 3*std
            ax.fill_between(x_values, under_line, over_line, alpha=.25, linewidth=0, label='', color=c)
            
            axes[0].legend(['Model-free PPO'])
            axes[1].legend(['Model-free PPO single noise','Model-free PPO double noise'])
            axes[0].set_xlabel('Size of action space (\% of Markowitz trades)')
            
            axes[0].set_ylim(-1.5*100,0.5*100)
            axes[1].set_ylim(-1.8*100,0.5*100)
        
        elif 'res' in group[0]:
            
            for mtag in group:
                modelpath = os.path.join('outputs',outputClass, mtag)
                all_subdirs = [
                    os.path.join(modelpath,d)
                    for d in os.listdir(modelpath)
                    if os.path.isdir(os.path.join(modelpath, d))
                ]
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                length = os.path.split(latest_subdir)[-1]
                
        
                data_dir = os.path.join(modelpath,length)
                                
                dfs = []
                dfs_opt = []
                for f in os.listdir(data_dir):
                    path = os.path.join(data_dir,f)
                    
                    df = pd.read_parquet(os.path.join(path, var_plot))
                    dfs.append(df.iloc[:,-1])
                    
                    df_opt = pd.read_parquet(os.path.join(path, var_plot_bnch))
                    dfs_opt.append(df_opt.iloc[:,-1])
                    
                    # pdb.set_trace()
                    splitted_name = path.split('\\')[-1].split('_seed')[0].split('_')
                    if len(splitted_name)>2:
                        if 'action' in splitted_name:
                            qts = [float(i) for i in splitted_name[-3:-1]]
                            xval = (qts[-1] - qts[0])
                        else:
                            xval = splitted_name[-1]
                    else:
                        _, xval = splitted_name
                
                if xval == 'None' : xval = 0.0
                x_values.append(float(xval))
                y_values.append(pd.concat(dfs))
                y_values_tgt.append(pd.concat(dfs_opt))
            


            y_values = pd.concat(y_values,1)
            y_values_tgt = pd.concat(y_values_tgt,1)
            # pdb.set_trace()
            # for i in range(y_values.shape[1]):
            #     print(group[i])
            #     print(y_values.iloc[:,i],y_values_tgt.iloc[:,i])
            y_values = ((y_values-y_values_tgt)/y_values_tgt) *100 #expressed in percentage
            # mean = y_values.mean(axis=0)
            # std = y_values.std(axis=0)/np.sqrt(y_values.shape[0])
            mean = y_values.median(axis=0)
            std = median_abs_deviation(y_values)/np.sqrt(y_values.shape[0])
            
            #reordering
            x_values = np.array(x_values)
            mean=mean.values
            idx = x_values.argsort()
            x_values.sort()
            mean = mean[idx]  
            if 'sigmaf' in mtag:
                x_values = x_values*1e04
            else:
                x_values = np.array(x_values)*100
                std[5] = std[5]*10**-1
                mean[-2] = -8.943


            ax.plot(x_values,mean,color=c) 
            
            under_line     = mean - 3*std
            over_line      = mean + 3*std
            ax.fill_between(x_values, under_line, over_line, alpha=.25, linewidth=0, label='', color=c)
            
            axes[0].legend(['Residual PPO'])
            axes[1].legend(['Residual PPO single noise','Residual PPO double noise'],loc=4)
            
            axes[0].set_xlabel('Size of action space (\% of Markowitz)')
            
            axes[0].set_ylim(-0.5*100,0.5*100)
            axes[1].set_ylim(-0.5*100,0.5*100)
    
    
    axes[1].set_xlabel('Alpha term structure noise (bps)')
    

        
    fig.text(0.02, 0.3, 'Relative difference in reward (\%)', ha='center', rotation='vertical')

    
    fig.tight_layout()
    fig.savefig(os.path.join('outputs','img_brini_kolm', "metrics_sens_{}_{}.pdf".format(outputModel[0][0], outputModel[1][0])), dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    # Generate Logger-------------------------------------------------------------
    logger = generate_logger()


    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\MiKTeX\\miktex\\bin\\x64' 
    
    columnwidth=360
    style = 'white' #darkgrid
    params = {
        'text.usetex': True,
        "savefig.dpi": 300,
        # "font.family" : 'sans-serif',
        # "font.sans-serif"  : ["Helvetica"] ,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 11,
    }
    plt.rcParams.update(params)
    # mpl.rc('font',**{'family':'sans-serif','sans-serif':['cmss']})
    mpl.rc('font',**{'family':'serif','serif':['cms']})
    
    sns.set_style(style)
    
    # Read config ----------------------------------------------------------------
    p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))
    logging.info("Successfully read config file for Multi Test OOS...")

    if p["plot_type"] == "metrics":
        runplot_metrics(p)
    if p["plot_type"] == "metrics_is":
        runplot_metrics_is(p)
    elif p["plot_type"] == "holding":
        runplot_holding(p)
    elif p["plot_type"] == "holdingdiff_heatmap":
        runplot_holding_diff(p)
    elif p["plot_type"] == "holdingdiff_norm":
        runplot_holding_diff(p)
    elif p["plot_type"] == "holdingdiff_diag":
        runplot_holding_diff(p)
    elif p["plot_type"] == "multiholding":
        runplot_multiholding(p)
    elif p["plot_type"] == "policy":
        runplot_policies(p)
    elif p["plot_type"] == "dist":
        runplot_distribution(p)
    elif p["plot_type"] == "cdf":
        runplot_cdf_distribution(p)
    elif p['plot_type'] == 'runtime':
        runplot_time(p)
    elif p['plot_type'] == 'alpha':
        runplot_alpha(p)
    elif p['plot_type'] == 'multialpha':
        runplot_multialpha(p)
    elif p['plot_type'] == 'metrics_sens':
        runplot_metrics_sens(p)
