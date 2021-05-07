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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec
import tensorflow as tf
import pdb
import seaborn as sns

sns.set_style("darkgrid")
import gin

gin.enter_interactive_mode()

from utils.plot import (
    plot_pct_metrics,
    plot_abs_metrics,
    plot_BestActions,
    plot_vf,
    load_DQNmodel,
    load_PPOmodel,
    plot_portfolio,
    plot_action,
)
from utils.test import Out_sample_vs_gp
from utils.env import MarketEnv
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

            # read main folder
            fig = plt.figure(figsize=set_size(width=1000.0))
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
                    
                    if 'Pdist' in v:
                        std = 1e+10
                        ax.set_ylim(0.0, 0.0 + std)
                    else:
                        value = dataframe_opt.iloc[0, 2]
                        std = 25000
                        ax.set_ylim(value - std, value + std)


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
                    ax.set_ylim(20, 220)
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

    fig2 = plt.figure(figsize=set_size(width=1000.0, subplots=(2, 2)))
    gs2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
    ax12 = fig2.add_subplot(gs2[0])
    ax22 = fig2.add_subplot(gs2[1])
    ax32 = fig2.add_subplot(gs2[2])
    ax42 = fig2.add_subplot(gs2[3])
    axes2 = [ax12, ax22, ax32, ax42]
    

    fig3 = plt.figure(figsize=set_size(width=1000.0, subplots=(2, 2)))
    gs3 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
    ax13 = fig3.add_subplot(gs3[0])
    ax23 = fig3.add_subplot(gs3[1])
    ax33 = fig3.add_subplot(gs3[2])
    ax43 = fig3.add_subplot(gs3[3])
    axes3 = [ax13, ax23, ax33, ax43]

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

        rng = np.random.RandomState(query("%SEED"))

        if query("%MV_RES"):
            action_space = ResActionSpace()
        else:
            action_space = ActionSpace()

        if query("%INP_TYPE") == "f":
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

        oos_test = Out_sample_vs_gp(
            savedpath=None,
            tag=tag[0],
            experiment_type=query("%EXPERIMENT_TYPE"),
            env_cls=MarketEnv,
            MV_res=query("%MV_RES"),
        )

        res_df = oos_test.run_test(train_agent, return_output=True)

        plot_portfolio(res_df, tag[0], axes[i])
        plot_action(res_df, tag[0], axes2[i])
        split = model.split("mv_res")
        axes[i].set_title(
            "_".join(["mv_res", split[-1]]).replace("_", " "), fontsize=10
        )
        axes2[i].set_title(
            "_".join(["mv_res", split[-1]]).replace("_", " "), fontsize=10
        )

        if '18' not in model.split('_')[0]:
            plot_action(res_df, tag[0], axes3[i], hist=True)
            axes3[i].set_title(
                "_".join(["mv_res", split[-1]]).replace("_", " "), fontsize=10
            )

    fig.suptitle('Holdings: ' + split[0].replace("_", " "))
    fig2.suptitle('Actions: ' + split[0].replace("_", " "))
    if '18' not in model.split('_')[0]:  
        fig2.suptitle('Res Actions: ' + split[0].replace("_", " "))
    


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
