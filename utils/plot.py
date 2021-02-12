# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:32:11 2020

@author: aless
"""
import numpy as np
from utils.DQN import DeepNetworkModel
from utils.DDPG import CriticNetwork, ActorNetwork
import os, sys, pdb
import pandas as pd
from tqdm import tqdm
from typing import Union, Optional
from utils.simulation import ReturnSampler, GARCHSampler, create_lstm_tensor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.env import MarketEnv, RecurrentMarketEnv, ReturnSpace, HoldingSpace, ActionSpace 
from utils.tools import CalculateLaggedSharpeRatio, RunModels
import collections
from natsort import natsorted
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib import cm
from utils.common import format_tousands
from utils.tools import get_bet_size


# LOAD UTILS
def load_DQNmodel(
    p: dict,
    data_dir: str,
    ckpt: bool = False,
    ckpt_it: int = None,
    ckpt_folder: str = None,
):
    """
    Load trained parameter for DQN

    Parameters
    ----------
    p: dict
        Parameter config loaded from the folder experiment

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
    num_inp = 2  # TODO insert number of factors as parameter
    if not p["zero_action"]:
        actions = np.arange(-p["KLM"][0], p["KLM"][0] + 1, p["KLM"][1])
        actions = actions[actions != 0]
    else:
        actions = np.arange(-p["KLM"][0], p["KLM"][0] + 1, p["KLM"][1])
    num_actions = len(actions)

    model = DeepNetworkModel(
        p["seed_init"],
        num_inp,
        p["hidden_units"],
        num_actions,
        p["batch_norm_input"],
        p["batch_norm_hidden"],
        p["activation"],
        p["kernel_initializer"],
        modelname="TrainNet",
    )

    if ckpt:
        if not ckpt_folder == "ckpt_pt":
            model.load_weights(
                os.path.join(data_dir, "ckpt", "DQN_{}_it_weights".format(ckpt_it))
            )
        else:
            model.load_weights(
                os.path.join(
                    data_dir, "ckpt_pt", "DQN_{}_it_pretrained_weights".format(ckpt_it)
                )
            )
    else:
        model.load_weights(os.path.join(data_dir, "DQN_final_weights"))

    return model, actions


class TrainedQTable:
    """
    Class which represents the Q-table to approximate the action-value function
    in the Q-learning algorithm. Used for inference since there are no update methods
    ...

    Attributes
    ----------
    Q_space : pd.DataFrame
        Current Q-table estimate

    Methods
    -------
    getQvalue(state: np.ndarray)-> np.ndarray
        Get the estimated action-value for each action at the current state

    argmaxQ(state: np.ndarray)-> int
        Get the index position of the action that gives the maximum action-value
        at the current state

    getMaxQ(state: np.ndarray)-> int
        Get the action that gives the maximum action-value at the current state

    chooseAction(state: np.ndarray, epsilon: float)-> int
        Get the index position of the action that gives the maximum action-value
        at the current state or a random action depeding on the epsilon parameter
        of exploration

    chooseGreedyAction(state: np.ndarray)-> int
        Get the index position of the action that gives the maximum action-value
        at the current state

    """
    def __init__(self, Q_space):
        # generate row index of the dataframe with every possible combination
        # of state space variables
        self.Q_space = Q_space

    def getQvalue(self, state):
        ret = state[0]
        holding = state[1]
        return self.Q_space.loc[
            (ret, holding),
        ]

    def argmaxQ(self, state):
        return self.getQvalue(state).idxmax()

    def getMaxQ(self, state):
        return self.getQvalue(state).max()

    def chooseAction(self, state, epsilon):
        random_action = self.rng.random()
        if random_action < epsilon:
            # pick one action at random for exploration purposes
            # dn = self.ActionSpace.sample()
            dn = self.rng.choice(self.ActionSpace.values)

        else:
            # pick the greedy action
            dn = self.argmaxQ(state)

        return dn

    def chooseGreedyAction(self, state):
        return self.argmaxQ(state)


def load_Actor_Critic(
    p, data_dir, ckpt=False, ckpt_it=None, ckpt_folder=None, DDPG_type="DDPG"
):
    num_states = 2
    num_actions = 1

    if ckpt:
        if not ckpt_folder == "ckpt_pt":
            if DDPG_type == "DDPG":
                # Q_model = CriticNetwork(p['seed_init'], num_states, p['hidden_units_Q'], num_actions,
                #                                p['batch_norm_input'], p['batch_norm_hidden'], p['activation'], p['kernel_initializer'],
                #                                p['output_init'], p['delayed_actions'],
                #                                modelname='Qmodel')
                p_model = ActorNetwork(
                    p["seed_init"],
                    num_states,
                    p["hidden_units_p"],
                    num_actions,
                    p["batch_norm_input"],
                    p["batch_norm_hidden"],
                    p["activation_p"],
                    p["kernel_initializer"],
                    p["output_init"],
                    modelname="pmodel",
                )
                # Q_model.load_weights(os.path.join(data_dir, 'ckpt','Q_model_{}_it_weights'.format(ckpt_it)))
                p_model.load_weights(
                    os.path.join(
                        data_dir, "ckpt", "p_model_{}_it_weights".format(ckpt_it)
                    )
                )
            elif DDPG_type == "TD3":
                # Q1_model = CriticNetwork(p['seed_init'], num_states, p['hidden_units_Q'], num_actions,
                #                                p['batch_norm_input'], p['batch_norm_hidden'], p['activation'], p['kernel_initializer'],
                #                                p['output_init'], p['delayed_actions'],
                #                                modelname='Q1model')
                # Q2_model = CriticNetwork(p['seed_init'], num_states, p['hidden_units_Q'], num_actions,
                #                                p['batch_norm_input'], p['batch_norm_hidden'], p['activation'], p['kernel_initializer'],
                #                                p['output_init'], p['delayed_actions'],
                #                                modelname='Q2model')
                p_model = ActorNetwork(
                    p["seed_init"],
                    num_states,
                    p["hidden_units_p"],
                    num_actions,
                    p["batch_norm_input"],
                    p["batch_norm_hidden"],
                    p["activation_p"],
                    p["kernel_initializer"],
                    p["output_init"],
                    modelname="pmodel",
                )
                # Q1_model.load_weights(os.path.join(data_dir, 'ckpt','Q1_model_{}_it_weights'.format(ckpt_it)))
                # Q2_model.load_weights(os.path.join(data_dir, 'ckpt','Q2_model_{}_it_weights'.format(ckpt_it)))
                p_model.load_weights(
                    os.path.join(
                        data_dir, "ckpt", "p_model_{}_it_weights".format(ckpt_it)
                    )
                )
        else:
            # Q_model.load_weights(os.path.join(data_dir, 'ckpt_pt','Q_model_{}_it_pretrained_weights'.format(ckpt_it)))
            p_model.load_weights(
                os.path.join(
                    data_dir,
                    "ckpt_pt",
                    "p_model_{}_it_pretrained_weights".format(ckpt_it),
                )
            )
    else:
        if DDPG_type == "DDPG":
            # Q_model = CriticNetwork(p['seed_init'], num_states, p['hidden_units_Q'], num_actions,
            #                                p['batch_norm_input'], p['batch_norm_hidden'], p['activation'], p['kernel_initializer'],
            #                                p['output_init'], p['delayed_actions'],
            #                                modelname='Qmodel')
            p_model = ActorNetwork(
                p["seed_init"],
                num_states,
                p["hidden_units_p"],
                num_actions,
                p["batch_norm_input"],
                p["batch_norm_hidden"],
                p["activation_p"],
                p["kernel_initializer"],
                p["output_init"],
                modelname="pmodel",
            )
            # Q_model.load_weights(os.path.join(data_dir, 'Q_model_final_weights'))
            p_model.load_weights(os.path.join(data_dir, "p_model_final_weights"))
        elif DDPG_type == "TD3":
            # Q1_model = CriticNetwork(p['seed_init'], num_states, p['hidden_units_Q'], num_actions,
            #                                p['batch_norm_input'], p['batch_norm_hidden'], p['activation'], p['kernel_initializer'],
            #                                p['output_init'], p['delayed_actions'],
            #                                modelname='Q1model')
            # Q2_model = CriticNetwork(p['seed_init'], num_states, p['hidden_units_Q'], num_actions,
            #                                p['batch_norm_input'], p['batch_norm_hidden'], p['activation'], p['kernel_initializer'],
            #                                p['output_init'], p['delayed_actions'],
            #                                modelname='Q2model')
            p_model = ActorNetwork(
                p["seed_init"],
                num_states,
                p["hidden_units_p"],
                num_actions,
                p["batch_norm_input"],
                p["batch_norm_hidden"],
                p["activation_p"],
                p["kernel_initializer"],
                p["output_init"],
                modelname="pmodel",
            )
            # Q1_model.load_weights(os.path.join(data_dir, 'Q1_model_final_weights'))
            # Q2_model.load_weights(os.path.join(data_dir, 'Q2_model_final_weights'))
            p_model.load_weights(os.path.join(data_dir, "p_model_final_weights"))

    return p_model
    # if DDPG_type == 'DDPG':
    #     return Q_model, p_model
    # elif DDPG_type == 'TD3':
    #     return Q1_model, Q2_model, p_model

# PLOT UTILS
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors



def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    # fig_height_in = fig_width_in * golden_ratio
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_multitest_overlap_OOS(
    ax1,
    df,
    data_dir,
    N_test,
    variable,
    colors=["b", "darkblue"],
    conf_interval=False,
    diff_colors=False,
    params=None,
    plot_lr=False,
    plot_experience=False,
    plot_buffer=False,
):

    df_mean = df.mean(axis=0)

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

    pgarch = np.round(
        np.array(
            [
                [
                    value
                    for key, value in params.items()
                    if "garch_omega" in key.lower()
                ],
                [
                    value
                    for key, value in params.items()
                    if "garch_alpha" in key.lower()
                ],
                [value for key, value in params.items() if "garch_beta" in key.lower()],
            ]
        ).ravel(),
        2,
    )
    # if pgarch.size!=0:
    #     lab  = 'sp_{}_om_{}_alpha_{},beta_{}_sum_{}'.format(params['seedparam'],pgarch[0],pgarch[1],pgarch[2],np.round(sum(pgarch[1:]),2))
    #     ax1.plot(idxs,df_mean.values,color=colors,linewidth=3, label='Avg {} {} {}'.format('DQN',variable.split('_')[0],
    #                                                                                           lab))

    # else:
    ax1.plot(
        idxs,
        df_mean.values,
        color=colors,
        linewidth=3,
        label="Avg {} {} {}".format(
            "DQN", variable.split("_")[0], data_dir.split("/")[-2]
        ),
    )

    if conf_interval:
        ci = 2 * np.std(df_mean.values)
        ax1.fill_between(
            idxs, (df_mean.values - ci), (df_mean.values + ci), color=colors, alpha=0.5
        )
    # add benchmark series to plot the hline

    if 'datatype' not in params.keys():
        params['datatype'] = 'gp'
    if pgarch.size == 0 and 'garch' not in params['datatype']:
        df.loc["Benchmark"] = 100.0
        ax1.plot(
            idxs, df.loc["Benchmark"].values, linestyle="--", linewidth=4, color="red"
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
            if params["datatype"] == "t_stud":
                ax1.set_ylim(-1500000, 1500000)
                # ax1.set_ylim(0.0,1.0)
            elif params["datatype"] == "garch":
                # ax1.set_ylim(-100000,100000)
                ax1.set_ylim(-1000000, 1000000)
            elif params["datatype"] == "garch_mr":
                # ax1.set_ylim(-100000,100000)
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
            # ax1.set_ylim(-500,500)

    ax1.set_title("{}".format(data_dir.split("/")[-2]))
    ax1.set_ylabel("% Reference {}".format(variable.split("_")[0]))
    ax1.set_xlabel("in-sample training iterations")

    ax1.legend()
    # scientific_formatter = FuncFormatter(scientific)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())

    if plot_lr:
        N = idxs[-1]
        rates = pd.DataFrame()
        initial_learning_rate = params["learning_rate"]
        decay_rate = params["exp_decay_rate"]
        decay_steps = params["exp_decay_pct"] * N
        l = lr_exp_decay(N, initial_learning_rate, decay_rate, decay_steps)
        rates[decay_steps / N] = l

        rates = rates / initial_learning_rate * 100
        idxs[-1] = idxs[-1] - 1

        if "exp_decay_pct" in data_dir or "exp_decay_rate" in data_dir:
            ax1.plot(
                idxs, rates.iloc[idxs], lw=2, linestyle="-", color=colors, alpha=0.8
            )
        else:
            ax1.plot(
                idxs, rates.iloc[idxs], lw=2, linestyle="-", color="blue", alpha=0.8
            )  # , label='lr_{}'.format(data_dir.split('/')[-2]))#

    if plot_experience:
        N = idxs[-1]
        epsilon = 1.0
        min_eps_pct = params["min_eps_pct"]
        min_eps = params["min_eps"]
        steps_to_min_eps = int(N * min_eps_pct)
        eps_decay = (epsilon - min_eps) / steps_to_min_eps
        e = eps(N, epsilon, eps_decay, min_eps)
        e = e / epsilon * 100
        idxs[-1] = idxs[-1] - 1

        if "min_eps_pct" in data_dir:
            ax1.plot(idxs, e[idxs], lw=2, linestyle="--", color=colors, alpha=0.8)
        else:
            ax1.plot(idxs, e[idxs], lw=2, linestyle="--", color="blue", alpha=0.8)

    if plot_buffer:
        N = idxs[-1]
        max_exp_pct = params["max_exp_pct"]

        if "max_exp_pct" in data_dir:
            ax1.axvline(
                x=max_exp_pct * N, lw=2, linestyle="-.", color=colors, alpha=0.8
            )
        else:
            ax1.axvline(
                x=max_exp_pct * N, lw=2, linestyle="-.", color="blue", alpha=0.8
            )

    # fig.savefig(os.path.join(data_dir,'{}.pdf'.format(variable)), dpi=300)


def lr_exp_decay(N, initial_learning_rate, decay_rate, decay_steps):

    lrs = []

    for i in range(N):
        lr = initial_learning_rate * decay_rate ** (i / decay_steps)
        lrs.append(lr)

    return np.array(lrs)


def eps(N, epsilon, eps_decay, min_eps):

    eps = []

    for i in range(N):
        epsilon = max(min_eps, epsilon - eps_decay)
        eps.append(epsilon)

    return np.array(eps)


def plot_abs_pnl_OOS(
    ax1, df, data_dir, N_test, variable, colors=["b", "darkblue"], params=None
):

    df_mean = df.mean(axis=0)
    
    
    idxs = [int(i) for i in df.iloc[0, :].index]

    for j, i in enumerate(df.index):
        ax1.scatter(x=idxs, y=df.iloc[i, :], alpha=0.6, color=colors, marker="o", s=7.5)
    ax1.plot(
        idxs,
        df_mean.values,
        color=colors,
        linewidth=3,
        label="{} {} {}".format(
            variable.split("_")[0], variable.split("_")[3], data_dir.split("/")[-2]
        ),
    )

    ax1.set_title("{}".format(data_dir.split("/")[-2]))
    ax1.set_ylabel("{}".format(variable.split("_")[0]))
    ax1.set_xlabel("in-sample training iterations")

    ax1.legend()
    # scientific_formatter = FuncFormatter(scientific)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylim(-20000000, 20000000)


def plot_multitest_real_OOS(
    ax1,
    df,
    data_dir,
    N_test,
    variable,
    colors=["b", "darkblue"],
    conf_interval=False,
    diff_colors=False,
    params=None,
    plot_lr=False,
    plot_experience=False,
    plot_buffer=False,
):

    df_mean = df.mean(axis=0)
    # pdb.set_trace()
    if params['training'] == 'offline':
        idxs = [int(i)* params['len_series'] for i in df.iloc[0, :].index]
    else:
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
        label="Avg {} {} {}".format(
            "DQN", variable.split("_")[0], data_dir.split("/")[-2]
        ),
    )

    if conf_interval:
        ci = 2 * np.std(df_mean.values)
        ax1.fill_between(
            idxs, (df_mean.values - ci), (df_mean.values + ci), color=colors, alpha=0.5
        )
    # add benchmark series to plot the hline


    if variable.split("_")[0] != "SR":
        df.loc["Benchmark"] = 0.0
        ax1.plot(
            idxs,
            df.loc["Benchmark"].values,
            linestyle="--",
            linewidth=4,
            color="red",
        )

        ax1.set_ylim(-2500, 2500)

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


    ax1.set_title("{}".format(data_dir.split("/")[-2]))
    ax1.set_ylabel("% Reference {}".format(variable.split("_")[0]))
    ax1.set_xlabel("in-sample training iterations")

    ax1.legend()
    # scientific_formatter = FuncFormatter(scientific)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())

    if plot_lr:
        N = idxs[-1]
        rates = pd.DataFrame()
        initial_learning_rate = params["learning_rate"]
        decay_rate = params["exp_decay_rate"]
        decay_steps = params["exp_decay_pct"] * N
        l = lr_exp_decay(N, initial_learning_rate, decay_rate, decay_steps)
        rates[decay_steps / N] = l

        rates = rates / initial_learning_rate * 100
        idxs[-1] = idxs[-1] - 1

        if "exp_decay_pct" in data_dir or "exp_decay_rate" in data_dir:
            ax1.plot(
                idxs, rates.iloc[idxs], lw=2, linestyle="-", color=colors, alpha=0.8
            )
        else:
            ax1.plot(
                idxs, rates.iloc[idxs], lw=2, linestyle="-", color="blue", alpha=0.8
            )  # , label='lr_{}'.format(data_dir.split('/')[-2]))#

    if plot_experience:
        N = idxs[-1]
        epsilon = 1.0
        min_eps_pct = params["min_eps_pct"]
        min_eps = params["min_eps"]
        steps_to_min_eps = int(N * min_eps_pct)
        eps_decay = (epsilon - min_eps) / steps_to_min_eps
        e = eps(N, epsilon, eps_decay, min_eps)
        e = e / epsilon * 100
        idxs[-1] = idxs[-1] - 1

        if "min_eps_pct" in data_dir:
            ax1.plot(idxs, e[idxs], lw=2, linestyle="--", color=colors, alpha=0.8)
        else:
            ax1.plot(idxs, e[idxs], lw=2, linestyle="--", color="blue", alpha=0.8)

    if plot_buffer:
        N = idxs[-1]
        max_exp_pct = params["max_exp_pct"]

        if "max_exp_pct" in data_dir:
            ax1.axvline(
                x=max_exp_pct * N, lw=2, linestyle="-.", color=colors, alpha=0.8
            )
        else:
            ax1.axvline(
                x=max_exp_pct * N, lw=2, linestyle="-.", color="blue", alpha=0.8
            )

    # fig.savefig(os.path.join(data_dir,'{}.pdf'.format(variable)), dpi=300)

# def scientific(x, pos):
#     # x:  tick value - ie. what you currently see in yticks
#     # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
#     return '%.2E' % x

########################################################################################################################
# Plot picture for the paper
def plot_multitest_paper(
    ax1,
    tag,
    df,
    data_dir,
    N_test,
    variable,
    colors=["b", "darkblue"],
    params=None,
    plt_bench=False,
):

    df_mean = df.mean(axis=0)

    idxs = [int(i) for i in df.iloc[0, :].index]
    for j, i in enumerate(df.index):
        ax1.scatter(
            x=idxs, y=df.iloc[i, :], alpha=0.15, color=colors, marker="o", s=0.5
        )

    pgarch = np.round(
        np.array(
            [
                [
                    value
                    for key, value in params.items()
                    if "garch_omega" in key.lower()
                ],
                [
                    value
                    for key, value in params.items()
                    if "garch_alpha" in key.lower()
                ],
                [value for key, value in params.items() if "garch_beta" in key.lower()],
            ]
        ).ravel(),
        2,
    )

    ax1.plot(idxs, df_mean.values, color=colors, linewidth=1.0, label=tag)

    # if 'datatype' in params:
    #     cond = params['datatype']!='t_stud'
    # else:
    #     cond = True
    cond = True
    # if pgarch.size==0 and params['datatype']!='t_stud':

    if plt_bench:
        if pgarch.size == 0 and cond:
            df.loc["Benchmark"] = 100.0
            ax1.plot(
                idxs,
                df.loc["Benchmark"].values,
                linestyle="--",
                linewidth=1.5,
                color="red",
                label="benchmark",
            )

            ax1.set_ylim(20, 130)
            # ax1.set_ylim(20,120)

        else:
            if variable.split("_")[0] != "SR":
                df.loc["Benchmark"] = 0.0
                ax1.plot(
                    idxs,
                    df.loc["Benchmark"].values,
                    linestyle="--",
                    linewidth=1.5,
                    color="red",
                )
                if params["datatype"] == "t_stud":
                    ax1.set_ylim(-1500000, 1500000)
                    # ax1.set_ylim(0.0,1.0)
                elif params["datatype"] == "garch":
                    # ax1.set_ylim(-100000,100000)

                    ax1.set_ylim(-1000000, 1000000)
                    # pass

            else:
                df.loc["Benchmark"] = 100.0
                ax1.plot(
                    idxs,
                    df.loc["Benchmark"].values,
                    linestyle="--",
                    linewidth=1.5,
                    color="red",
                )
                ax1.set_ylim(20, 130)
                # ax1.set_ylim(-500,500)

