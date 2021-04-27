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
from utils.math_tools import unscale_action
# from utils.env import MarketEnv
# from utils.tools import CalculateLaggedSharpeRatio, RunModels
import collections
from natsort import natsorted
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib import cm
import torch
from agents.PPO import PPOActorCritic

# # LOAD UTILS
def load_DQNmodel(
    data_dir: str,
    ckpt_it: int = None,
    model: object = None,
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
        
        if query('%INP_TYPE')=='f': 
            num_inp = 3
        else:
            num_inp = 2
    
        if query('%MV_RES'):
            actions = ResActionSpace(query('%ACTION_RANGE_RES'), query('%ZERO_ACTION')).values
        else:
            actions = ActionSpace(query('%ACTION_RANGE'), query('%ZERO_ACTION'), query('%SIDE_ONLY')).values
                
        num_actions = len(actions)
    
        model = DeepNetworkModel(
            query('%SEED'),
            num_inp,
            query('DQN.hidden_units'),
            num_actions,
            query('DQN.batch_norm_input'),
            query('DQN.batch_norm_hidden'),
            query('DQN.activation'),
            query('DQN.kernel_initializer'),
            modelname="TrainNet",
        )

        model.load_weights(
            os.path.join(data_dir, "ckpt", "DQN_{}_ep_weights".format(ckpt_it))
        )
        
        return model, actions
    
    else:
        model.load_weights(
            os.path.join(data_dir, "ckpt", "DQN_{}_ep_weights".format(ckpt_it))
        )
        
        return model

    


def load_PPOmodel(   data_dir: str,
    ckpt_it: int = None,
    model: object = None,):
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
        
        if query('%INP_TYPE')=='f': 
            inp_shape = (3,)
        else:
            inp_shape = (2,)
    
        if query('%MV_RES'):
            actions = ResActionSpace(query('%ACTION_RANGE_RES'), query('%ZERO_ACTION')).values
        else:
            actions = ActionSpace(query('%ACTION_RANGE'), query('%ZERO_ACTION'), query('%SIDE_ONLY')).values

        num_actions = actions.ndim
    
        model = PPOActorCritic(
            query('%SEED'),
            inp_shape,
            query('PPO.activation'),
            query('PPO.hidden_units_value'),
            query('PPO.hidden_units_actor'),
            num_actions,
            query('PPO.batch_norm_input'),
            query('PPO.batch_norm_value_out'),
            query('PPO.policy_type'),
            query('PPO.init_pol_std'),
            query('PPO.min_pol_std'),
            query('PPO.std_transform'),
            query('PPO.init_last_layers'),
            modelname="PPO",
        )

        model.load_state_dict(
            torch.load(
                os.path.join(data_dir, "ckpt", "PPO_{}_ep_weights.pth".format(ckpt_it))
            )
        )
        
        return model, actions
        
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
        label='_'.join(data_dir.split('/')[-2].split('_')[2:]
        ),
    )

    if conf_interval:
        ci = 2 * np.std(df_mean.values)
        ax1.fill_between(
            idxs, (df_mean.values - ci), (df_mean.values + ci), color=colors, alpha=0.5
        )
        

    if gin.query_parameter('%DATATYPE') != 'garch':

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
            ax1.set_ylim(0, 150)
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
            if gin.query_parameter('%DATATYPE') == "t_stud":
                ax1.set_ylim(-1500000, 1500000)
            elif gin.query_parameter('%DATATYPE') == "garch":
                ax1.set_ylim(-1000000, 1000000)
            elif gin.query_parameter('%DATATYPE') == "garch_mr":
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
            ax1.set_ylim(0, 150)

    ax1.set_title("{}: {} simulation".format(variable.split("_")[3].split('.')[0],data_dir.split('_')[1]))
    ax1.set_ylabel("% Reference {}".format(variable.split("_")[0]))
    ax1.set_xlabel("in-sample training iterations")

    ax1.legend()
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())

    # fig.savefig(os.path.join(data_dir,'{}.pdf'.format(variable)), dpi=300)



def plot_abs_metrics(
    ax1, df, df_opt, data_dir, N_test, variable, colors=["b", "darkblue"], i=0
):


    # df_mean = df.mean(axis=0)
    # df_opt = df_opt.mean(axis=0)
    df_mean = df.median(axis=0)
    df_opt = df_opt.median(axis=0)

    idxs = [int(i) for i in df.iloc[0, :].index]

    for j, i in enumerate(df.index):
        ax1.scatter(x=idxs, y=df.iloc[i, :], alpha=0.6, color=colors, marker="o", s=7.5)
    ax1.plot(
        idxs,
        df_mean.values,
        color=colors,
        linewidth=3,
        label="{}".format(
             '_'.join(data_dir.split('/')[-2].split('_')[2:])
        ),
    )
    
    ax1.plot(
        idxs,
        df_opt.values,
        color='red',
        linewidth=3,
        label="GP" if i == 0 else "",
    )

    ax1.set_title("{}: {} simulation".format(variable.split("_")[3].split('.')[0],data_dir.split('_')[1]))
    ax1.set_ylabel("{}".format(variable.split("_")[0]))
    ax1.set_xlabel("in-sample training iterations")

    ax1.legend()
    # ax1.set_ylim(-20000000, 20000000)

    # fig.savefig(os.path.join(data_dir,'{}.pdf'.format(variable)), dpi=300)

def plot_vf(
    model,
    actions: list,
    holding: float,
    ax: object = None,
    less_labels: bool = False,
    n_less_labels: int = None,
    optimal=False
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
    
    if query('%INP_TYPE') == 'ret':
        sample_Ret = np.linspace(-0.05, 0.05, 100)
        
        if holding == 0:
            holdings = np.zeros(len(sample_Ret), dtype='float')
        else:
            holdings = np.ones(len(sample_Ret), dtype='float') * holding

        if model.modelname == 'DQN':
            states = tf.constant(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1))),
                dtype=tf.float32,
            )
            pred = model(states, training=False)
        
        elif model.modelname == 'PPO':
            states = torch.from_numpy(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1)))).float()
            with torch.no_grad():
                _, pred = model(states)
            

    elif query('%INP_TYPE') == 'f':
        
        n_factors = len(query('%F_PARAM'))
        
        f_to_concat = [np.linspace(-0.5 - i, 0.5 + i, 100).reshape(-1,1) for i,_ in enumerate(range(n_factors))]
        
        factors = np.concatenate(f_to_concat,axis=1, dtype='float')
        
        if holding == 0:
            holdings = np.zeros(len(factors), dtype='float')
        else:
            holdings = np.ones(len(factors), dtype='float') * holding
            

        if model.modelname == 'DQN':
            states = tf.constant(
                np.hstack((factors, holdings.reshape(-1, 1))),
                dtype=tf.float32,
            )
    
            pred = model(states, training=False)
        
        elif model.modelname == 'PPO':
            states = torch.from_numpy(
                np.hstack((factors, holdings.reshape(-1, 1)))).float()
            with torch.no_grad():
                _, pred = model(states)

    viridis = cm.get_cmap("viridis", pred.shape[1])
    for i in range(pred.shape[1]):
        if less_labels:
            subset_label_idx = np.round(
                np.linspace(0, len(actions) - 1, n_less_labels)
            ).astype(int)
            subset_label = actions[subset_label_idx]
            if actions[i] in subset_label:
                ax.plot(
                    states[:, 0],
                    pred[:, i],
                    label=str(actions[i]),
                    c=viridis.colors[i],
                    linewidth=1.5,
                )
            else:
                ax.plot(
                    states[:, 0],
                    pred[:, i],
                    label="_nolegend_",
                    c=viridis.colors[i],
                    linewidth=1.5,
                )
        else:
            ax.plot(
                states[:, 0],
                pred[:, i],
                label=str(actions[i]),
                c=viridis.colors[i],
                linewidth=1.5,
            )
            
    # if optimal and query('%INP_TYPE') == 'f':
    
    discount_rate, kappa, costmultiplier, f_param, halflife, sigma = (query('%DISCOUNT_RATE'), query('%KAPPA'),
    query('%COSTMULTIPLIER'), query('%F_PARAM'), query('%HALFLIFE'), query('%SIGMA'))
    
    n_factors = len(query('%F_PARAM'))
    
    f_to_concat = [np.linspace(-0.5 - i, 0.5 + i, 100).reshape(-1,1) for i,_ in enumerate(range(n_factors))]
    
    factors = np.concatenate(f_to_concat,axis=1)
    
    
    V = optimal_vf(states, discount_rate, kappa, costmultiplier, f_param, halflife, sigma)

    ax.plot(factors[:,0], V, linewidth=1.5, label='GP Vf')
            
def optimal_vf(states, discount_rate, kappa, costmultiplier, f_param, halflife, sigma):
    
    def opt_trading_rate_disc_loads(discount_rate, kappa, CostMultiplier, f_param, f_speed):
    
        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-discount_rate / 260)
    
        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = kappa * (1 - rho) + CostMultiplier * rho
        num2 = np.sqrt(
            num1 ** 2 + 4 * kappa * CostMultiplier * (1 - rho) ** 2
        )
        den = 2 * (1 - rho)
        a = (-num1 + num2) / den
    
        OptRate = a / CostMultiplier
        DiscFactorLoads = f_param / (
            1 + f_speed * ((OptRate * CostMultiplier) / kappa)
        )
    
        return OptRate, DiscFactorLoads
    f_speed = np.around(np.log(2) / halflife, 4)
    OptRate, DiscFactorLoads = opt_trading_rate_disc_loads(discount_rate,
                                                       kappa,
                                                       costmultiplier, 
                                                       f_param, 
                                                       f_speed)
    

    disc_rate_bar = 1 - discount_rate
    lambda_bar = (costmultiplier*sigma**2)/disc_rate_bar
    costmultiplier_bar = costmultiplier/disc_rate_bar
    
    axx1 = (disc_rate_bar * kappa * lambda_bar * sigma**2)
    axx2 = 0.25 * (discount_rate**2 * lambda_bar**2 + 2 * discount_rate * kappa * lambda_bar * sigma**2 + \
                   (kappa**2 * lambda_bar * sigma**2)/lambda_bar)
    axx3 = - 0.5 * (discount_rate*lambda_bar + kappa * sigma**2)
    Axx = np.sqrt(axx1 + axx2) + axx3
    
    axf1 = disc_rate_bar/(1 - disc_rate_bar*(1-f_speed) * (1-Axx/lambda_bar))
    axf2 = (1-Axx/lambda_bar)*np.array(f_param)
    Axf = axf1 * axf2
    
    aff1 = disc_rate_bar/(1 - disc_rate_bar*(1-f_speed) * (1-f_speed))
    q = (np.array(f_param) + Axf*(1-f_speed)) ** 2 / (kappa*sigma**2 + lambda_bar + Axx)
    Aff = aff1 * q
    
    # v1 = - 0.5 * states[:,-1]**2 * Axx
    # v2 = np.dot(states[:,-1]*Axf, states[:,:-1])
    v3 = 0.5 * states[:,:-1] * Aff
    
    
    # V = v1 + v2 + v3
    V =  v3
    
    return V


def plot_BestActions(
    model, holding: float, ax: object = None, optimal: bool = False,
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
    
    def opt_trading_rate_disc_loads(discount_rate, kappa, CostMultiplier, f_param, f_speed):
    
        # 1 percent annualized discount rate (same rate of Ritter)
        rho = 1 - np.exp(-discount_rate / 260)
    
        # kappa is the risk aversion, CostMultiplier the parameter for trading cost
        num1 = kappa * (1 - rho) + CostMultiplier * rho
        num2 = np.sqrt(
            num1 ** 2 + 4 * kappa * CostMultiplier * (1 - rho) ** 2
        )
        den = 2 * (1 - rho)
        a = (-num1 + num2) / den
    
        OptRate = a / CostMultiplier
        DiscFactorLoads = f_param / (
            1 + f_speed * ((OptRate * CostMultiplier) / kappa)
        )
    
        return OptRate, DiscFactorLoads
    
    if query('%MV_RES'):
        actions = ResActionSpace(query('%ACTION_RANGE_RES'), query('%ZERO_ACTION')).values
    else:
        actions = ActionSpace(query('%ACTION_RANGE'), query('%ZERO_ACTION'), query('%SIDE_ONLY')).values


    if query('%INP_TYPE') == 'ret':
        sample_Ret = np.linspace(-0.05, 0.05, 100, dtype='float')
        
        if holding == 0:
            holdings = np.zeros(len(sample_Ret), dtype='float')
        else:
            holdings = np.ones(len(sample_Ret), dtype='float') * holding
                    
        if model.modelname == 'DQN':
            states = tf.constant(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1))),
                dtype=tf.float32,
            )
            pred = model(states, training=False)
        
            max_action = actions[tf.math.argmax(pred, axis=1)]
        elif model.modelname == 'PPO':
            states = torch.from_numpy(
                np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1)))).float()
            with torch.no_grad():
                dist, _ = model(states)

            unscaled_max_action = torch.nn.Tanh()(dist.mean)
            scaled_max_action = unscale_action(
                actions[-1], unscaled_max_action
            )

        
        
        ax.plot(sample_Ret, scaled_max_action, linewidth=1.5, label='{} Policy'.format(model.modelname))
        
        
    elif query('%INP_TYPE') == 'f':
        
        n_factors = len(query('%F_PARAM'))
        
        f_to_concat = [np.linspace(-0.5 - i, 0.5 + i, 100).reshape(-1,1) for i,_ in enumerate(range(n_factors))]
        
        factors = np.concatenate(f_to_concat,axis=1, dtype='float')
        
        if holding == 0:
            holdings = np.zeros(len(factors), dtype='float')
        else:
            holdings = np.ones(len(factors), dtype='float') * holding

        if model.modelname == 'DQN':
            states = tf.constant(
                np.hstack((factors, holdings.reshape(-1, 1))),
                dtype=tf.float32,
            )
            
            pred = model(states, training=False)
        
            max_action = actions[tf.math.argmax(pred, axis=1)]
        elif model.modelname == 'PPO':
            states = torch.from_numpy(
                np.hstack((factors, holdings.reshape(-1, 1)))).float()
            with torch.no_grad():
                dist, _ = model(states)

            unscaled_max_action = torch.nn.Tanh()(dist.mean)
            scaled_max_action = unscale_action(
                actions[-1], unscaled_max_action
            )


        ax.plot(factors[:,0], max_action, linewidth=1.5, label='{} Policy'.format(model.modelname))


    if optimal:
        
        discount_rate, kappa, costmultiplier, f_param, halflife, sigma = (query('%DISCOUNT_RATE'), query('%KAPPA'),
        query('%COSTMULTIPLIER'), query('%F_PARAM'), query('%HALFLIFE'), query('%SIGMA'))
        
        OptRate, DiscFactorLoads = opt_trading_rate_disc_loads(discount_rate,
                                                               kappa,
                                                               costmultiplier, 
                                                               f_param, 
                                                               np.around(np.log(2) / halflife, 4))

        if query('%INP_TYPE') == 'ret':
            
            OptNextHolding = (1 - OptRate) * holding + OptRate * (
                                    1 / (kappa * (sigma) ** 2)
                                )  * sample_Ret
            optimal_policy = OptNextHolding - holding
            
            ax.plot(sample_Ret, optimal_policy, linewidth=1.5, label='GP Policy')
        elif query('%INP_TYPE') == 'f':
            
            OptNextHolding = (1 - OptRate) * holding + OptRate * (
                                    1 / (kappa * (sigma) ** 2)
                                ) * np.sum(DiscFactorLoads * factors, axis=1)
            optimal_policy = OptNextHolding - holding

            ax.plot(factors[:,0], optimal_policy, linewidth=1.5, label='GP Policy')
            
            


def plot_portfolio(r: pd.DataFrame, tag: str, ax2: object):
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

    ax2.plot(r["NextHolding_{}".format(tag)].values[1:-1], label=tag)
    ax2.plot(r["OptNextHolding"].values[1:-1], label="benchmark", alpha=0.5)
    
def plot_action(r: pd.DataFrame, tag: str, ax2: object):
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

    ax2.plot(r["Action_{}".format(tag)].values[1:-1], label=tag)
    ax2.plot(r["OptNextAction"].values[1:-1], label="benchmark", alpha=0.5)