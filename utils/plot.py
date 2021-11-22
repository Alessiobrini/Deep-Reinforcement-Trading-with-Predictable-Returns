# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:32:11 2020

@author: aless
"""
import gin

gin.enter_interactive_mode()
import numpy as np
import tensorflow as tf
from agents.DQN import DeepNetworkModel
import os, pdb
import pandas as pd
from utils.spaces import (
    ActionSpace,
    ResActionSpace,
)
from utils.common import set_size
from utils.math_tools import unscale_action, unscale_asymmetric_action
from utils.simulation import DataHandler
# from utils.env import MarketEnv
# from utils.tools import CalculateLaggedSharpeRatio, RunModels
import collections
from natsort import natsorted
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import torch
from agents.PPO import PPOActorCritic
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import median_abs_deviation

# # LOAD UTILS
def load_DQNmodel(
    data_dir: str, ckpt_it: int = None, model: object = None,
):
    """
    Load trained parameter for DQN

    Parameters
    ----------

    data_dir: str
        Dicretory model where the weights are store

    ckpt: bool
        Boolean to regulate if the loaded weights are a checkpoint or not

    ckpt_it: int
        Number of iteration of the checkpoint you want to load

    ckpt_folder: bool
        boolean if you want to load weights from pretrained models

    Returns
    ----------
    model
        Model with loaded weights
    actions: np.ndarray
        Array of possible actions

    """
    if not model:
        query = gin.query_parameter

        if query("%INP_TYPE") == "f":
            num_inp = len(query('%F_PARAM')) + 1
        else:
            num_inp = 2

        if query("%MV_RES"):
            actions = ResActionSpace(
                query("%ACTION_RANGE_RES"), query("%ZERO_ACTION")
            ).values
        else:
            actions = ActionSpace(
                query("%ACTION_RANGE"), query("%ZERO_ACTION"), query("%SIDE_ONLY")
            ).values

        num_actions = len(actions)

        model = DeepNetworkModel(
            query("%SEED"),
            num_inp,
            query("DQN.hidden_units"),
            num_actions,
            query("DQN.batch_norm_input"),
            query("DQN.batch_norm_hidden"),
            query("DQN.activation"),
            query("DQN.kernel_initializer"),
            modelname="TrainNet",
        )

        model.load_weights(
            os.path.join(data_dir, "ckpt", "DQN_{}_ep_weights".format(ckpt_it))
        )
        
        model.modelname = 'DQN'
        
        return model, actions

    else:
        model.load_weights(
            os.path.join(data_dir, "ckpt", "DQN_{}_ep_weights".format(ckpt_it))
        )
        
        model.modelname = 'DQN'

        return model


def load_PPOmodel(
    data_dir: str, ckpt_it: int = None, model: object = None,
):
    """
    Load trained parameter for DQN

    Parameters
    ----------
    d: dict
        Parameter config loaded from the folder experiment

    data_dir: str
        Dicretory model where the weights are stores

    ckpt: bool
        Boolean to regulate if the loaded weights are a checkpoint or not

    ckpt_it: int
        Number of iteration of the checkpoint you want to load

    Returns
    ----------
    model
        Model with loaded weights
    actions: np.ndarray
        Array of possible actions

    """
    if not model:
        query = gin.query_parameter
        if gin.query_parameter('%MULTIASSET'):
            n_assets = len(gin.query_parameter('%HALFLIFE'))
            n_factors = len(gin.query_parameter('%HALFLIFE')[0])
            if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                inp_shape = (n_factors*n_assets+n_assets+1,1)
            else:
                inp_shape = (n_assets+n_assets+1,1)
        else:
            if query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":
                inp_shape = (len(query('%F_PARAM')) + 1,)
            else:
                inp_shape = (2,)

        if query("%MV_RES"):
            actions = ResActionSpace(
                query("%ACTION_RANGE_RES"), query("%ZERO_ACTION")
            ).values
        else:
            actions = ActionSpace(
                query("%ACTION_RANGE"), query("%ZERO_ACTION"), query("%SIDE_ONLY")
            ).values

        if gin.query_parameter('%MULTIASSET'):
            num_actions = len(gin.query_parameter('%HALFLIFE'))
        else:
            num_actions = actions.ndim

        model = PPOActorCritic(
            query("%SEED"),
            inp_shape,
            query("PPO.activation"),
            query("PPO.hidden_units_value"),
            query("PPO.hidden_units_actor"),
            num_actions,
            query("PPO.batch_norm_input"),
            query("PPO.batch_norm_value_out"),
            query("PPO.policy_type"),
            query("PPO.init_pol_std"),
            query("PPO.min_pol_std"),
            query("PPO.std_transform"),
            query("PPO.init_last_layers"),
            modelname="PPO",
        )

        model.load_state_dict(
            torch.load(
                os.path.join(data_dir, "ckpt", "PPO_{}_ep_weights.pth".format(ckpt_it))
            )
        )

        return model, actions

    else:
        if ckpt_it == 'rnd':
            pass
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(data_dir, "ckpt", "PPO_{}_ep_weights.pth".format(ckpt_it))
                )
            )

    return model


def plot_pct_metrics(
    ax1,
    df,
    data_dir,
    N_test,
    variable,
    colors=["b", "darkblue"],
    conf_interval=False,
    diff_colors=False,
    params_path=None,
):

    gin.parse_config_file(params_path, skip_unknown=True)

    # df_mean = df.mean(axis=0)
    df_mean = df.median(axis=0)

    idxs = [int(i) for i in df.iloc[0, :].index]
    # https://matplotlib.org/examples/color/colormaps_reference.html
    colormap = cm.get_cmap("plasma", len(df.index))
    for j, i in enumerate(df.index):
        if diff_colors:
            ax1.scatter(
                x=idxs,
                y=df.iloc[i, :],
                alpha=0.6,
                color=colormap.colors[j],
                marker="o",
                s=7.5,
            )
        else:
            ax1.scatter(
                x=idxs, y=df.iloc[i, :], alpha=0.6, color=colors, marker="o", s=7.5
            )

    ax1.plot(
        idxs,
        df_mean.values,
        color=colors,
        linewidth=3,
        label="_".join(data_dir.split("/")[-2].split("_")[2:]),
    )

    if conf_interval:
        ci = 2 * np.std(df_mean.values)
        ax1.fill_between(
            idxs, (df_mean.values - ci), (df_mean.values + ci), color=colors, alpha=0.5
        )

    if gin.query_parameter("%DATATYPE") != "garch":

        if variable.split("_")[0] == "Pdist":

            ax1.set_ylim(-1e4, 1e12)
        else:

            df.loc["Benchmark"] = 100.0
            ax1.plot(
                idxs,
                df.loc["Benchmark"].values,
                linestyle="--",
                linewidth=4,
                color="red",
            )
            # ax1.set_ylim(-10000,300)
            # ax1.set_ylim(0, 150)
            # ax1.set_ylim(-150,150)

    else:
        if variable.split("_")[0] != "SR":
            df.loc["Benchmark"] = 0.0
            ax1.plot(
                idxs,
                df.loc["Benchmark"].values,
                linestyle="--",
                linewidth=4,
                color="red",
            )
            if gin.query_parameter("%DATATYPE") == "t_stud":
                ax1.set_ylim(-1500000, 1500000)
            elif gin.query_parameter("%DATATYPE") == "garch":
                # ax1.set_ylim(-1000000, 1000000)
                pass
            elif gin.query_parameter("%DATATYPE") == "garch_mr":
                ax1.set_ylim(-5000000, 5000000)

        else:
            df.loc["Benchmark"] = 100.0
            ax1.plot(
                idxs,
                df.loc["Benchmark"].values,
                linestyle="--",
                linewidth=4,
                color="red",
            )
            # ax1.set_ylim(0, 150)

    ax1.set_title(
        "{}: {} simulation".format(
            variable.split("_")[3].split(".")[0], data_dir.split("_")[1]
        )
    )
    ax1.set_ylabel("% Reference {}".format(variable.split("_")[0]))
    ax1.set_xlabel("in-sample training iterations")

    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())

    # fig.savefig(os.path.join(data_dir,'{}.pdf'.format(variable)), dpi=300)


def plot_abs_metrics(
    ax1, df, df_opt, data_dir, N_test, variable, colors=["b", "darkblue"], i=0, plt_type='diff'
):

    if plt_type == 'diff':

        idxmax = df.mean(1).idxmax()
        select_agent = 'best'
        if select_agent == 'mean':
            ppo = df.mean(0)
            gp = df_opt.mean(0)
        elif select_agent == 'median':
            ppo = df.median(0)
            gp = df_opt.median(0)
        elif select_agent == 'best':
            ppo = df.loc[idxmax]
            gp = df_opt.loc[idxmax]


        reldiff_avg = (ppo-gp)/gp * 100

            
        # df = ((df-df_opt)/df_opt)*100
        # df_median = df.median(axis=0)
        # mad = median_abs_deviation(df)
        # if 'Pdist' not in variable:
        #     df_opt = df_opt.median(axis=0)
        

        idxs = [int(i) for i in reldiff_avg.index]
        ax1.plot(
            idxs,
            reldiff_avg.values,
            color=colors,
            linewidth=2,
            label="{}".format("_".join(data_dir.split("/")[-2].split("_")[2:])),
        )
        # sz=1.0
        # under_line     = df_median - sz *mad
        # over_line      = df_median + sz *mad
        # ax1.fill_between(idxs, under_line, over_line, color=colors, alpha=0.25, linewidth=0, label='')

    elif plt_type == 'abs':


        df_median = df.median(axis=0)

        if 'Pdist' not in variable:
            df_opt = df_opt.median(axis=0)
        
    
        idxs = [int(i) for i in df.iloc[0, :].index]
    
        for j, i in enumerate(df.index):
            ax1.scatter(x=idxs, y=df.iloc[i, :], alpha=0.6, color=colors, marker="o", s=3.5)
        ax1.plot(
            idxs,
            df_median.values,
            color=colors,
            linewidth=2,
            label="{}".format("_".join(data_dir.split("/")[-2].split("_")[2:])),
        )
        if 'Pdist' not in variable:
            ax1.plot(
                idxs, df_opt.values, color="red", linestyle= '--', linewidth=2, label="GP" if i == 0 else "",
            )


def plot_vf(
    model,
    actions: list,
    holding: float,
    ax: object = None,
    less_labels: bool = False,
    n_less_labels: int = None,
    optimal=False,
):

    """
    Ploduce plots of learned action-value function of DQN

    Parameters
    ----------
    model
        Loaded model

    actions: list
        List of possible action for the mdel

    holding: float
        Fixed holding at which we produce the value function plot

    ax: matplotlib.axes.Axes
        Axes to draw in

    less_labels: bool
        Boolean to regulate if all labels appear in the legend. If False, they all appear

    n_less_labels: int
        Number of labels to include in the legend

    """

    query = gin.query_parameter

    if query("%INP_TYPE") == "ret" or query("%INP_TYPE") == "alpha":
        if query("%INP_TYPE") == "alpha":
            data_handler = DataHandler(N_train=query('%LEN_SERIES'), rng=None)
            data_handler.generate_returns()
            sample_Ret = data_handler.returns
            sample_Ret.sort()
        else:
            sample_Ret = np.linspace(-0.05, 0.05, query('%LEN_SERIES'), dtype="float")

        if holding == 0:
            holdings = np.zeros(len(sample_Ret), dtype="float")
        else:
            holdings = np.ones(len(sample_Ret), dtype="float") * holding

        if model.modelname == "DQN":
            states = tf.constant(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1))),
                dtype=tf.float32,
            )
            pred = model(states, training=False)

        elif model.modelname == "PPO":
            states = torch.from_numpy(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1)))
            ).float()
            with torch.no_grad():
                _, pred = model(states)

    elif query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":

        if query("%INP_TYPE") == "alpha_f":
            data_handler = DataHandler(N_train=query('%LEN_SERIES'), rng=None)
            data_handler.generate_returns()
            factors = data_handler.factors
            factors.sort()
        else:
            n_factors = len(query("%F_PARAM"))
    
            f_to_concat = [
                np.linspace(-0.5 - i, 0.5 + i, query('%LEN_SERIES')).reshape(-1, 1)
                for i, _ in enumerate(range(n_factors))
            ]
    
            factors = np.concatenate(f_to_concat, axis=1)

        if holding == 0:
            holdings = np.zeros(len(factors), dtype="float")
        else:
            holdings = np.ones(len(factors), dtype="float") * holding

        if model.modelname == "DQN":
            states = tf.constant(
                np.hstack((factors, holdings.reshape(-1, 1))), dtype=tf.float32,
            )

            pred = model(states, training=False)

        elif model.modelname == "PPO":
            states = torch.from_numpy(
                np.hstack((factors, holdings.reshape(-1, 1)))
            ).float()
            with torch.no_grad():
                _, pred = model(states)

    for i in range(pred.shape[1]):
        ax.plot(
            states[:, 0],
            pred[:, i],
            label='PPO Vf',
            linewidth=1.5,
        )

    if optimal and (query('%INP_TYPE') == 'f' or query('%INP_TYPE') == 'alpha_f'):

        discount_rate, kappa, costmultiplier, f_param, halflife, sigma = (
            query("%DISCOUNT_RATE"),
            query("%KAPPA"),
            query("%COSTMULTIPLIER"),
            query("%F_PARAM"),
            query("%HALFLIFE"),
            query("%SIGMA"),
        )


        V = optimal_vf(
            states, discount_rate, kappa, costmultiplier, f_param, halflife, sigma
        )
        pdb.set_trace()
        ax.plot(factors[:, 0], V, linewidth=1.5, label="GP Vf")

    ax.legend()


def optimal_vf(states, discount_rate, kappa, costmultiplier, f_param, halflife, sigma):
    def opt_trading_rate_disc_loads(
        discount_rate, kappa, CostMultiplier, f_param, f_speed
    ):

        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-discount_rate / 260)

        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = kappa * (1 - rho) + CostMultiplier * rho
        num2 = np.sqrt(num1 ** 2 + 4 * kappa * CostMultiplier * (1 - rho) ** 2)
        den = 2 * (1 - rho)
        a = (-num1 + num2) / den

        OptRate = a / CostMultiplier
        DiscFactorLoads = f_param / (1 + f_speed * ((OptRate * CostMultiplier) / kappa))

        return OptRate, DiscFactorLoads

    f_speed = np.around(np.log(2) / halflife, 4)
    OptRate, DiscFactorLoads = opt_trading_rate_disc_loads(
        discount_rate, kappa, costmultiplier, f_param, f_speed
    )

    disc_rate_bar = 1 - discount_rate
    lambda_bar = (costmultiplier * sigma ** 2) / disc_rate_bar
    costmultiplier_bar = costmultiplier / disc_rate_bar

    axx1 = disc_rate_bar * kappa * lambda_bar * sigma ** 2
    axx2 = 0.25 * (
        discount_rate ** 2 * lambda_bar ** 2
        + 2 * discount_rate * kappa * lambda_bar * sigma ** 2
        + (kappa ** 2 * lambda_bar * sigma ** 2) / lambda_bar
    )
    axx3 = -0.5 * (discount_rate * lambda_bar + kappa * sigma ** 2)
    Axx = np.sqrt(axx1 + axx2) + axx3

    axf1 = disc_rate_bar / (1 - disc_rate_bar * (1 - f_speed) * (1 - Axx / lambda_bar))
    axf2 = (1 - Axx / lambda_bar) * np.array(f_param)
    Axf = axf1 * axf2

    aff1 = disc_rate_bar / (1 - disc_rate_bar * (1 - f_speed) * (1 - f_speed))
    q = (np.array(f_param) + Axf * (1 - f_speed)) ** 2 / (
        kappa * sigma ** 2 + lambda_bar + Axx
    )
    Aff = aff1.reshape(-1,1) @ q.reshape(-1,1).T
    
    states = states.numpy()
    
    v1 = - 0.5 * states[:,-1]**2 * Axx
    v2s = []
    for i in range(states.shape[0]):
        v2 = states[i,-1].reshape(-1,1).T @ Axf.reshape(-1,1).T @ states[i,:-1].reshape(-1,1)
        v2s.append(v2)
    v2 = np.array(v2s).ravel()
    v3s = []
    for i in range(states.shape[0]):
        v3 = states[i,:-1].reshape(-1,1).T @ Aff @ states[i,:-1].reshape(-1,1)
        v3s.append(v3)
    v3 = 0.5 * np.array(v3s).ravel()

    # pdb.set_trace()
    V = v1 + v2 + v3


    return V


def plot_BestActions(
    model, holding: float, ax: object = None, optimal: bool = False, 
    stochastic:bool = True, seed: int = 324345, color='tab:blue', generate_plot=False
):

    """
    Ploduce plots of learned action-value function of DQN

    Parameters
    ----------
    p: dict
        Parameter passed as config files

    model
        Loaded model

    holding: float
        Fixed holding at which we produce the value function plot

    ax: matplotlib.axes.Axes
        Axes to draw in

    """

    query = gin.query_parameter
    # gin.bind_parameter('%DOUBLE_NOISE', False)
    # gin.bind_parameter('%SIGMAF', [None])
    # gin.bind_parameter('%INITIAL_ALPHA', [0.009])
    # gin.bind_parameter('%HALFLIFE', [35])
    gin.bind_parameter('alpha_term_structure_sampler.generate_plot', generate_plot)

    def opt_trading_rate_disc_loads(
        discount_rate, kappa, CostMultiplier, f_param, f_speed
    ):

        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-discount_rate / 260)

        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = kappa * (1 - rho) + CostMultiplier * rho
        num2 = np.sqrt(num1 ** 2 + 4 * kappa * CostMultiplier * (1 - rho) ** 2)
        den = 2 * (1 - rho)
        a = (-num1 + num2) / den

        OptRate = a / CostMultiplier
        DiscFactorLoads = f_param / (1 + f_speed * ((OptRate * CostMultiplier) / kappa))

        return OptRate, DiscFactorLoads

    if query("%MV_RES"):
        actions = ResActionSpace(
            query("%ACTION_RANGE_RES"), query("%ZERO_ACTION")
        ).values
    else:
        actions = ActionSpace(
            query("%ACTION_RANGE"), query("%ZERO_ACTION"), query("%SIDE_ONLY")
        ).values
        
    rng = np.random.RandomState(seed)
    if query("%INP_TYPE") == "ret" or query("%INP_TYPE") == "alpha":
        if query("%INP_TYPE") == "alpha":
            data_handler = DataHandler(N_train=query('%LEN_SERIES'), rng=rng)
            data_handler.generate_returns()
            sample_Ret = data_handler.returns
            sample_Ret.sort()
        else:
            sample_Ret = np.linspace(-0.05, 0.05, query('%LEN_SERIES'), dtype="float")

        if holding == 0:
            holdings = np.zeros(len(sample_Ret), dtype="float")
        else:
            holdings = np.ones(len(sample_Ret), dtype="float") * holding
            

        if model.modelname == "DQN":
            states = tf.constant(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1))),
                dtype=tf.float32,
            )
            pred = model(states, training=False)

            max_action = actions[tf.math.argmax(pred, axis=1)]
            
        elif model.modelname == "PPO":
            states = torch.from_numpy(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1)))
            ).float()
            with torch.no_grad():
                dist, _ = model(states)
            
            if stochastic:
                unscaled_max_action = torch.nn.Tanh()(dist.sample())
            else:
                unscaled_max_action = torch.nn.Tanh()(dist.mean)
                
            if query("%MV_RES"):
                max_action = unscale_asymmetric_action(actions[0],actions[-1], unscaled_max_action)
            else:
                max_action = unscale_action(actions[-1], unscaled_max_action)


        # ci = 1.96 * model.log_std.exp().detach().numpy()
        ax.plot(
            sample_Ret,
            max_action,
            linewidth=1.5,
            label="{} Policy".format(model.modelname),
            color=color
        )
        # ax.fill_between(sample_Ret, (max_action-ci).reshape(-1), (max_action+ci).reshape(-1), color='b', alpha=.1)

    
    elif query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":

        if query("%INP_TYPE") == "alpha_f":
            data_handler = DataHandler(N_train=query('%LEN_SERIES'), rng=rng)
            data_handler.generate_returns()
            factors = data_handler.factors
            
            # factors.sort()
        else:
            n_factors = len(query("%F_PARAM"))
    
            f_to_concat = [
                np.linspace(-0.5 - i, 0.5 + i, query('%LEN_SERIES')).reshape(-1, 1)
                for i, _ in enumerate(range(n_factors))
            ]
    
            factors = np.concatenate(f_to_concat, axis=1)

        if holding == 0:
            holdings = np.zeros(len(factors), dtype="float")
        elif holding == None:
            holdings = np.linspace(-1e+5, 1e+5, len(factors), dtype="float")
            if factors.shape[1]>1:
                factors = np.array([np.repeat(0.004,factors.shape[1])]*len(factors))
            else:
                factors = np.repeat(0.004, len(factors)).reshape(-1, 1)
        else:
            holdings = np.ones(len(factors), dtype="float") * holding


        if model.modelname == "DQN":
            states = tf.constant(
                np.hstack((factors, holdings.reshape(-1, 1))), dtype=tf.float32,
            )

            pred = model(states, training=False)

            max_action = actions[tf.math.argmax(pred, axis=1)]
        elif model.modelname == "PPO":
            states = torch.from_numpy(
                np.hstack((factors, holdings.reshape(-1, 1)))
            ).float()
            with torch.no_grad():
                dist, _ = model(states)

            if stochastic:
                unscaled_max_action = torch.nn.Tanh()(dist.sample())
            else:
                unscaled_max_action = torch.nn.Tanh()(dist.mean)

            if query("%MV_RES"):
                max_action = unscale_asymmetric_action(actions[0],actions[-1], unscaled_max_action).numpy().reshape(-1,)
            else:
                max_action = unscale_action(actions[-1], unscaled_max_action).numpy().reshape(-1,)
            

        if query("%MV_RES"):
            discount_rate, kappa, costmultiplier, f_param, halflife, sigma = (
                query("%DISCOUNT_RATE"),
                query("%KAPPA"),
                query("%COSTMULTIPLIER"),
                query("%F_PARAM"),
                query("%HALFLIFE"),
                query("%SIGMA"),
            )
    
            OptRate, DiscFactorLoads = opt_trading_rate_disc_loads(
                discount_rate,
                kappa,
                costmultiplier,
                f_param,
                np.around(np.log(2) / halflife, 4),
            )

            OptNextHolding = (1 / (kappa * (sigma) ** 2)) * np.sum(
                f_param * factors, axis=1
            )
            # Compute optimal markovitz action
            MV_action = OptNextHolding - holdings

            max_action = MV_action * (1 - max_action)
            
        if holding == None:
            ax.plot(
                holdings,
                max_action,
                linewidth=1.5,
                label="{} Policy".format(model.modelname),
                color=color
            )
        else:
            ax.plot(
                factors[:, 0]*10**4, #to express in bps
                max_action,
                linewidth=1.5,
                label="{} Policy".format(model.modelname),
                color=color
            )

        # ci = 1.96 * model.log_std.exp().detach().numpy()
        # ax.fill_between(factors[:, 0], (max_action-ci*max_action).reshape(-1), (max_action+ci*max_action).reshape(-1), color='b', alpha=.1)

    if optimal:

        discount_rate, kappa, costmultiplier, f_param, halflife, sigma = (
            query("%DISCOUNT_RATE"),
            query("%KAPPA"),
            query("%COSTMULTIPLIER"),
            query("%F_PARAM"),
            query("%HALFLIFE"),
            query("%SIGMA"),
        )

        OptRate, DiscFactorLoads = opt_trading_rate_disc_loads(
            discount_rate,
            kappa,
            costmultiplier,
            f_param,
            np.around(np.log(2) / halflife, 4),
        )

        if query("%INP_TYPE") == "ret" or query("%INP_TYPE") == "alpha":

            OptNextHolding = (1 - OptRate) * holding + OptRate * (
                1 / (kappa * (sigma) ** 2)
            ) * sample_Ret
            optimal_policy = OptNextHolding - holding

            ax.plot(sample_Ret, optimal_policy, linewidth=1.5, label="GP Policy")
        elif query("%INP_TYPE") == "f" or query("%INP_TYPE") == "alpha_f":

            OptNextHolding = (1 - OptRate) * holdings + OptRate * (
                1 / (kappa * (sigma) ** 2)
            ) * np.sum(DiscFactorLoads * factors, axis=1)
            optimal_policy = OptNextHolding - holdings

            if holding == None:
                ax.plot(holdings, optimal_policy, linewidth=1.5, label="GP Policy", color='tab:orange')
            else:
                ax.plot(factors[:, 0]*10**4, optimal_policy, linewidth=1.5, label="GP Policy", color='tab:orange')
                
            OptNextHolding_mv = (1 / (kappa * (sigma) ** 2)) * np.sum(
                f_param * factors, axis=1
            )
            # Compute optimal markovitz action
            MV_policy = OptNextHolding_mv - holdings

            if holding == None:
                ax.plot(holdings, MV_policy, linewidth=1.5, label="MV Policy", color='black')
            else:
                ax.plot(factors[:, 0]*10**4, MV_policy, linewidth=1.5, label="MV Policy", color='black')


def plot_portfolio(r: pd.DataFrame, tag: str, ax2: object, tbox: bool = True,colors: list = ['tab:blue','tab:orange']):
    """
    Ploduce plots of portfolio holding

    Parameters
    ----------
    r: pd.DataFrame
        Dataframe containing the variables

    tag: str
        Name of the algorithm to plot result

    ax2: matplotlib.axes.Axes
        Axes to draw in

    """

    if gin.query_parameter('%MULTIASSET'):
        ax2.plot(r.filter(like="NextHolding_{}".format(tag)).values[1:-1])
        ax2.plot(r.filter(like="OptNextHolding").values[1:-1], alpha=0.65, ls='--')

        
        n_lines = r.filter(like="NextHolding_{}".format(tag)).shape[-1]
        for i in range(n_lines):
            ax2.lines[-1-i].set_color(ax2.lines[n_lines-1-i].get_color())
            
        ax2.legend(list(r.filter(like="NextHolding_{}".format(tag)).columns) + 
                   list(r.filter(like="OptNextHolding").columns),fontsize=9)
        
        if tbox:
            hold_diff = r.filter(like="OptNextHolding").values[1:-1] -  r.filter(like="NextHolding_{}".format(tag)).values[1:-1] 
            norm = np.linalg.norm(hold_diff)
            norm_text = AnchoredText("Norm diff: {:e}".format(norm),loc= 'upper left',prop=dict(size=10))
            ax2.add_artist(norm_text)

    else:
        
        ax2.plot(r["OptNextHolding"].values[1:-1], label="benchmark", color=colors[1], ls='--')
        ax2.plot(r["NextHolding_{}".format(tag)].values[1:-1], label=tag, color=colors[0])
        if tbox:
            mse = np.round(np.sum(r["OptNextHolding"].values[1:-1] - r["NextHolding_{}".format(tag)].values[1:-1]),decimals=0)
            mse_text = AnchoredText("GP - PPO: {:e}".format(mse),loc=1,prop=dict(size=10))
            ax2.add_artist(mse_text)

def plot_2asset_holding(r: pd.DataFrame, tag: str, ax2: object):
    """
    Ploduce plots of portfolio holding

    Parameters
    ----------
    r: pd.DataFrame
        Dataframe containing the variables

    tag: str
        Name of the algorithm to plot result

    ax2: matplotlib.axes.Axes
        Axes to draw in

    """

    # ax2.hist(r.filter(like="NextHolding_{}".format(tag)).values[1:-1], edgecolor = 'black',color=['b','y'])
    # ax2.hist(r.filter(like='OptNextHolding').values[1:-1], alpha=0.65,color=['b','y'])
    
    sns.kdeplot(r.filter(like="NextHolding_{}".format(tag)).iloc[:,0].values[1:-1], bw_method=0.2,ax=ax2,color='b')
    sns.kdeplot(r.filter(like="NextHolding_{}".format(tag)).iloc[:,1].values[1:-1], bw_method=0.2,ax=ax2,color='y')
    sns.kdeplot(r.filter(like='OptNextHolding').iloc[:,0].values[1:-1], bw_method=0.2,ax=ax2,color='b', alpha=0.5, ls='--')
    sns.kdeplot(r.filter(like='OptNextHolding').iloc[:,1].values[1:-1], bw_method=0.2,ax=ax2,color='y', alpha=0.5, ls='--')

    ax2.set_xlabel('Position amount')
    ax2.set_ylabel('Frequency')
        

    ax2.legend(list(r.filter(like="NextHolding_{}".format(tag)).columns) + 
               list(r.filter(like="OptNextHolding").columns),fontsize=9)

def plot_heatmap_holding(r: pd.DataFrame, tag: str, title):
    """
    Ploduce plots of portfolio holding

    Parameters
    ----------
    r: pd.DataFrame
        Dataframe containing the variables

    tag: str
        Name of the algorithm to plot result

    ax2: matplotlib.axes.Axes
        Axes to draw in

    """

    # pdb.set_trace()
    fig,ax = plt.subplots(figsize=set_size(width=1000.0))
    
    hold_diff = r.filter(like="OptNextHolding").values[1:-1] -  r.filter(like="NextHolding_{}".format(tag)).values[1:-1] 
    scaler = MinMaxScaler((-1,1))
    hold_diff = scaler.fit_transform(hold_diff)

    sns.heatmap(hold_diff, ax=ax, cmap='viridis')
    
    ax.set_title(title,fontsize=9)
    
    ax.set_xlabel('Assets')
    ax.set_ylabel('Time')
        



def plot_action(r: pd.DataFrame, tag: str, ax2: object, hist=False):
    """
    Ploduce plots of portfolio holding

    Parameters
    ----------
    r: pd.DataFrame
        Dataframe containing the variables

    tag: str
        Name of the algorithm to plot result

    ax2: matplotlib.axes.Axes
        Axes to draw in

    """
    if hist:
        if "ResAction_{}".format(tag) in r.columns:
            ax2.hist(r["ResAction_{}".format(tag)].values[1:-1], label=tag)
        else:
            ax2.hist(r["Action_{}".format(tag)].values[1:-1], label=tag)
            ax2.hist(r["OptNextAction"].values[1:-1], label="benchmark", alpha=0.5)
    else:
        ax2.plot(r["Action_{}".format(tag)].values[1:-1], label=tag)
        ax2.plot(r["OptNextAction"].values[1:-1], label="benchmark", alpha=0.5)


def plot_costs(r: pd.DataFrame, tag: str, ax2: object, hist=False):
    """
    Ploduce plots of portfolio holding

    Parameters
    ----------
    r: pd.DataFrame
        Dataframe containing the variables

    tag: str
        Name of the algorithm to plot result

    ax2: matplotlib.axes.Axes
        Axes to draw in

    """
    if hist:
        ax2.hist(r["Cost_{}".format(tag)].values[1:-1], label=tag)
        ax2.hist(r["OptCost"].values[1:-1], label="benchmark", alpha=0.5)
    else:
        ax2.plot(r["Cost_{}".format(tag)].cumsum().values[1:-1], label=tag)
        ax2.plot(r["OptCost"].cumsum().values[1:-1], label="benchmark", alpha=0.5)
        
def move_sn_x(offs=0, dig=0, side='left', omit_last=False):
    """Move scientific notation exponent from top to the side.
    
    Additionally, one can set the number of digits after the comma
    for the y-ticks, hence if it should state 1, 1.0, 1.00 and so forth.

    Parameters
    ----------
    offs : float, optional; <0>
        Horizontal movement additional to default.
    dig : int, optional; <0>
        Number of decimals after the comma.
    side : string, optional; {<'left'>, 'right'}
        To choose the side of the y-axis notation.
    omit_last : bool, optional; <False>
        If True, the top y-axis-label is omitted.

    Returns
    -------
    locs : list
        List of y-tick locations.

    Note
    ----
    This is kind of a non-satisfying hack, which should be handled more
    properly. But it works. Functions to look at for a better implementation:
    ax.ticklabel_format
    ax.yaxis.major.formatter.set_offset_string
    """

    # Get the ticks
    locs, _ = plt.xticks()
    # pdb.set_trace()

    # Put the last entry into a string, ensuring it is in scientific notation
    # E.g: 123456789 => '1.235e+08'
    llocs = '%.3e' % locs[-1]

    # Get the magnitude, hence the number after the 'e'
    # E.g: '1.235e+08' => 8
    yoff = int(str(llocs).split('e')[1])


    # If omit_last, remove last entry
    if omit_last:
        slocs = locs[:-1]
    else:
        slocs = locs

    # Set ticks to the requested precision
    form = r'$%.'+str(dig)+'f$'

    plt.xticks(locs, list(map(lambda x: form % x, slocs/(10**yoff))))

    # Define offset depending on the side
    if side == 'left':
        offs = -.18 - offs # Default left: -0.18
    elif side == 'right':
        offs = 1 + offs    # Default right: 1.0

    # Plot the exponent
    plt.text(offs, 0.05, r'$\times10^{%i}$' % yoff, transform =
            plt.gca().transAxes, verticalalignment='top',fontsize=11)

    # Return the locs
    return locs
