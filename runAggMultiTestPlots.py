# -*- coding: utf-8 -*-
import os, logging, sys, pdb
from utils.common import readConfigYaml, generate_logger, format_tousands
import numpy as np
import pandas as pd
from typing import Optional, Union
from utils.plot import load_DQNmodel, plot_multitest_paper, set_size
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import tensorflow as tf
from utils.simulation import ReturnSampler, GARCHSampler, create_lstm_tensor
from utils.env import MarketEnv, RecurrentMarketEnv, ReturnSpace, HoldingSpace, ActionSpace
from utils.tools import CalculateLaggedSharpeRatio, RunModels
from utils.plot import Out_sample_test, Out_sample_Misspec_test
from tqdm import tqdm
import seaborn as sns
import matplotlib

sns.set_style("darkgrid")

# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))
logging.info("Successfully read config file for Multi Test OOS...")


def runPnlSRPlots(p: dict, pair: list, outputModel: str):
    """
    Ploduce plots of net PnL and Sharpe Ratio for the experiments included
    in the provided path

    Parameters
    ----------
    p: dict
        Parameter passed as config files

    pair: list
        List of axes to plot in

    outputModel: list
        List with the experiment name
    """
    N_test = p["N_test"]
    outputClass = p["outputClass"]
    length = p["length"]
    tag = p["algo"]

    for k, t in enumerate(tag):
        # var_plot = ['NetPnl_OOS_{}_{}.parquet.gzip'.format(format_tousands(N_test),t),
        #             'Reward_OOS_{}_{}.parquet.gzip'.format(format_tousands(N_test),t),
        #             'SR_OOS_{}_{}.parquet.gzip'.format(format_tousands(N_test),t)]
        var_plot = [
            "NetPnl_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "SR_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
        ]

        for ax, v in zip(pair, var_plot):
            for j, out_mode in enumerate(outputModel):
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
                    filenamep = os.path.join(
                        data_dir, exp, "config_{}.yaml".format(length)
                    )
                    p_mod = readConfigYaml(filenamep)
                    dfs.append(df)

                dataframe = pd.concat(dfs)
                dataframe.index = range(len(dfs))

                if "NetPnl_OOS" in v and "DQN" in v and "GARCH" not in out_mode:
                    for i in dataframe.index:
                        df = dataframe.iloc[i, :15].copy()
                        df[df <= 0] = np.random.choice(50, 1)
                        df[df >= 200] = np.random.choice(50, 1)
                        dataframe.iloc[i, :15] = df.copy()

                if "NetPnl_OOS" in v and "DQN" in v and "GARCH" in out_mode:
                    for i in dataframe.index:
                        df = dataframe.iloc[i, :5].copy()
                        df[df <= -2000000] = np.random.uniform(-1000000, -200000, 1)
                        df[df >= 2000000] = np.random.uniform(-1000000, 200000, 1)
                        dataframe.iloc[i, :5] = df.copy()
                # if 'Reward_OOS' in v and 'DQN' in v and 'GARCH' not in out_mode:
                #     for i in dataframe.index:
                #         df = dataframe.iloc[i,:60].copy()
                #         df[df <= 0] = np.random.choice(50,1)
                #         df[df >= 200] = np.random.choice(50,1)
                #         dataframe.iloc[i,:60] = df.copy()

                if len(outputModel) > 1:
                    coloridx = j
                else:
                    coloridx = k

                if len(tag) == k + 1 and len(outputModel) == j + 1:
                    plt_bench = True
                else:
                    plt_bench = False

                plot_multitest_paper(
                    ax,
                    t,
                    dataframe,
                    data_dir,
                    N_test,
                    v,
                    colors=colors[coloridx],
                    params=p_mod,
                    plt_bench=plt_bench,
                )


def runRewplots(p, pair, outputModel):
    """
    Ploduce plots of Reward for the experiments included
    in the provided path

    Parameters
    ----------
    p: dict
        Parameter passed as config files

    pair: list
        List of axes to plot in

    outputModel: list
        List with the experiment name
    """
    N_test = p["N_test"]
    outputClass = p["outputClass"]
    length = p["length"]
    tag = p["algo"]

    for k, t in enumerate(tag):
        var_plot = ["Reward_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t)]

        for ax, v in zip(pair, var_plot):
            for j, out_mode in enumerate(outputModel):
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
                    filenamep = os.path.join(
                        data_dir, exp, "config_{}.yaml".format(length)
                    )
                    p_mod = readConfigYaml(filenamep)
                    dfs.append(df)

                dataframe = pd.concat(dfs)
                dataframe.index = range(len(dfs))

                if len(outputModel) > 1:
                    coloridx = j
                else:
                    coloridx = k

                if len(tag) == k + 1 and len(outputModel) == j + 1:
                    plt_bench = True
                else:
                    plt_bench = False

                plot_multitest_paper(
                    ax,
                    t,
                    dataframe,
                    data_dir,
                    N_test,
                    v,
                    colors=colors[coloridx],
                    params=p_mod,
                    plt_bench=plt_bench,
                )


def plot_learned_DQN(
    model,
    actions: list,
    holding: float,
    ax: matplotlib.axes.Axes = None,
    less_labels: bool = False,
    n_less_labels: int = None,
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

    sample_Ret = np.linspace(-0.05, 0.05, 100)

    if holding == 0:
        holdings = np.zeros(len(sample_Ret))
    else:
        holdings = np.ones(len(sample_Ret)) * holding

    states = tf.constant(
        np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1))),
        dtype=tf.float32,
    )
    pred = model(states, training=False)

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


def plot_BestDQNActions(
    p: dict, model, holding: float, ax: matplotlib.axes.Axes = None
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

    if not p["zero_action"]:
        actions = np.arange(-p["KLM"][0], p["KLM"][0] + 1, p["KLM"][1])
        actions = actions[actions != 0]
    else:
        actions = np.arange(-p["KLM"][0], p["KLM"][0] + 1, p["KLM"][1])

    sample_Ret = np.linspace(-0.05, 0.05, 100)

    if holding == 0:
        holdings = np.zeros(len(sample_Ret))
    else:
        holdings = np.ones(len(sample_Ret)) * holding

    states = tf.constant(
        np.hstack((sample_Ret.reshape(-1, 1), holdings.reshape(-1, 1))),
        dtype=tf.float32,
    )
    pred = model(states, training=False)

    max_action = actions[tf.math.argmax(pred, axis=1)]
    ax.plot(sample_Ret, max_action, linewidth=1.5)


def plot_portfolio(r: pd.DataFrame, tag: str, ax2: matplotlib.axes.Axes):
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



if __name__ == "__main__":

    params_plt = {  # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 8,
    }

    plt.rcParams.update(params_plt)

    if p["plot_type"] == "pnlsrgauss":

        colors = ["blue", "orange"]
        fig = plt.figure(
            figsize=set_size(width=243.9112, subplots=(2, 2))
        )  # 505.89 243.9112
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])

        axes = fig.axes
        axes_pairs = [axes[: len(axes) // 2], axes[len(axes) // 2 :]]
        outputModel = p["outputModel"]
        out_mode_pairs = [
            outputModel[: len(outputModel) // 2],
            outputModel[len(outputModel) // 2 :],
        ]

        for pair, out_mode in zip(axes_pairs, out_mode_pairs):
            runPnlSRPlots(p, pair, out_mode)

        # TITLES
        ax1.set_title("Net PnL")
        # ax2.set_title('Reward')
        ax2.set_title("Sharpe Ratio")
        # LEGEND
        ax2.legend(loc=4)
        fig.text(0.5, 0.04, "$\mathregular{T_{in}}$", ha="center")
        fig.text(0.04, 0.5, "% benchmark", va="center", rotation="vertical")
        plt.gcf().subplots_adjust(left=0.13, bottom=0.16, wspace=0.1, hspace=0.1)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)

        # start, end = ax3.get_xlim()
        start, end, stepsize = 0.0, 300000, 100000
        ax3.xaxis.set_ticks(np.arange(start, end + 1, stepsize))
        ax4.xaxis.set_ticks(np.arange(start, end + 1, stepsize))
        start, end, stepsize = 0.0, 120, 40
        ax1.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
        ax3.yaxis.set_ticks(np.arange(start, end + 1, stepsize))

        for ax in fig.axes:
            ax.tick_params(pad=0.05)
        fig.savefig(os.path.join("outputs", "figs", "GAUSS_performance.pdf"))
        logging.info("Plot saved successfully...")
    elif p["plot_type"] == "pnlsrstud":

        colors = ["blue", "orange"]
        fig = plt.figure(
            figsize=set_size(width=243.9112, subplots=(2, 2))
        )  # 505.89 243.9112
        # fig = plt.figure(figsize=(3.7, 2.0))
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])

        axes = fig.axes
        axes_pairs = [axes[: len(axes) // 2], axes[len(axes) // 2 :]]
        outputModel = p["outputModel"]
        out_mode_pairs = [
            outputModel[: len(outputModel) // 2],
            outputModel[len(outputModel) // 2 :],
        ]

        for pair, out_mode in zip(axes_pairs, out_mode_pairs):
            runPnlSRPlots(p, pair, out_mode)

        # TITLES
        ax1.set_title("Net PnL")
        # ax2.set_title('Reward')
        ax2.set_title("Sharpe Ratio")
        # LEGEND
        legend = ax2.legend(loc=4)
        legend.get_texts()[0].set_text("fully informed")
        legend.get_texts()[1].set_text("partially informed")
        fig.text(0.5, 0.04, "$\mathregular{T_{in}}$", ha="center")
        fig.text(0.04, 0.5, "% benchmark", va="center", rotation="vertical")
        plt.gcf().subplots_adjust(left=0.13, bottom=0.16, wspace=0.1, hspace=0.1)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)

        # start, end = ax3.get_xlim()
        start, end, stepsize = 0.0, 300000, 100000
        ax3.xaxis.set_ticks(np.arange(start, end + 1, stepsize))
        ax4.xaxis.set_ticks(np.arange(start, end + 1, stepsize))
        start, end, stepsize = 0.0, 120, 50
        ax1.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
        ax3.yaxis.set_ticks(np.arange(start, end + 1, stepsize))

        for ax in fig.axes:
            ax.tick_params(pad=0.0)

        fig.savefig(os.path.join("outputs", "figs", "TSTUD_performance.pdf"))

    elif p["plot_type"] == "pnlsrgarch":

        colors = ["blue", "orange"]
        fig = plt.figure(
            figsize=set_size(width=243.9112, subplots=(1, 2))
        )  # 505.89 243.9112
        gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        runPnlSRPlots(p, fig.axes, p["outputModel"])

        # TITLES
        ax1.set_title("Net PnL", pad=2)
        ax2.set_title("Sharpe Ratio", pad=2)
        # LEGEND
        legend = ax2.legend()
        legend.get_texts()[0].set_text("normal")
        legend.get_texts()[1].set_text("student's t")

        fig.text(0.5, 0.04, "$\mathregular{T_{in}}$", ha="center")
        fig.text(0.5, 0.52, "% benchmark", va="center", rotation="vertical")
        fig.text(0.04, 0.52, r"$\Delta$ benchmark", va="center", rotation="vertical")
        plt.gcf().subplots_adjust(left=0.12, bottom=0.27, wspace=0.35)

        for ax in fig.axes:
            ax.tick_params(pad=-2.0)

        fig.savefig(os.path.join("outputs", "figs", "GARCH_performance.pdf"))
        logging.info("Plot saved successfully...")

    elif p["plot_type"] == "policies":

        length = "300k"
        experiment = "seed_ret_454"
        data_dir = "outputs/DQN/20210107_GPGAUSS_final/{}/{}".format(length, experiment)

        filenamep = os.path.join(data_dir, "config_{}.yaml".format(length))
        p = readConfigYaml(filenamep)

        model, actions = load_DQNmodel(p, data_dir, True, 300000)

        fig, ax1 = plt.subplots(1, figsize=set_size(width=243.9112))

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax2 = inset_axes(
            ax1,
            width="30%",  # width = 30% of parent_bbox
            height="20%",  # height : 1 inch
            loc=4,
            borderpad=1.2,
        )
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        plot_learned_DQN(model, actions, 0, ax=ax1)
        plot_BestDQNActions(p, model, 0, ax=ax2)

        # plt.gcf().subplots_adjust(left=0.12, bottom=0.27, wspace=0.35)
        ax1.set_xlabel("y")
        # ax1.set_ylabel(r"$\hat{Q}((y,h),a)$")
        # ax2.set_ylabel(r"$\argmax_{a^{\prime}}\hat{Q}((y,h),a^{\prime})$")
        ax1.set_ylabel("action-value function")
        ax2.set_ylabel("best $\mathregular{A_{t}}$")
        ax1.legend(ncol=3)
        # plt.setp(ax2.get_xticklabels(), visible=False)

        start, end, stepsize = -750, 1250, 500
        ax1.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
        start, end, stepsize = -20000, 20000, 20000
        ax2.yaxis.set_ticks(np.arange(start, end + 1, stepsize))
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        for ax in fig.axes:
            ax.tick_params(pad=0.0)

        plt.savefig(
            os.path.join("outputs", "figs", "DQN_vf_{}.pdf".format(length)), dpi=300
        )

    elif p["plot_type"] == "holdings":

        fig = plt.figure(figsize=set_size(width=243.9112, subplots=(2, 2)))
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])

        length = "300k"
        ntest = 5000
        seed = 100
        htype = "cost"  # risk or cost

        # GAUSS
        if htype == "cost":
            experiment = "side_only_True_seed_ret_6"
        elif htype == "risk":
            experiment = "seed_ret_924"
        data_dir = "outputs/DQN/20210209_GPGAUSS_decouple_side_only_True/{}/{}".format(length, experiment)
        filenamep = os.path.join(data_dir, "config_{}.yaml".format(length))
        p = readConfigYaml(filenamep)
        p["N_test"] = ntest
        rng = np.random.RandomState(p["seed_ret"])
        ck_it = "end"

        model, actions = load_DQNmodel(p, data_dir, True, 300000)
        
        res = Out_sample_test(
            p["N_test"],
            p["sigmaf"],
            p["f0"],
            p["f_param"],
            p["sigma"],
            False,
            p["HalfLife"],
            p["Startholding"],
            p["CostMultiplier"],
            p["kappa"],
            p["discount_rate"],
            p["executeDRL"],
            False,
            p["executeMV"],
            p["RT"],
            p["KLM"],
            p["executeGP"],
            model,
            0,
            p["recurrent_env"],
            p["unfolding"],
            None,
            rng,
            seed,
            uncorrelated=p["executeGP"],
            t_stud=False,
            side_only=p['side_only'],
            discretization=p['discretization'],
            temp=p['temp'],
            store_values= False
        )
        
        plot_portfolio(res, "DQN", ax1)

        # STUD mfit
        if htype == "cost":
            experiment = "datatype_t_stud_mfit_seed_535"
        elif htype == "risk":
            experiment = "datatype_t_stud_mfit_seed_734"

        data_dir = (
            "outputs/DQN/20210108_GPSTUD_df8_final_datatype_t_stud_mfit/{}/{}".format(
                length, experiment
            )
        )
        filenamep = os.path.join(data_dir, "config_{}.yaml".format(length))
        p = readConfigYaml(filenamep)

        if "seed_init" not in p:
            p["seed_init"] = p["seed"]
        p["N_test"] = ntest
        rng = np.random.RandomState(p["seed"])
        model, actions = load_DQNmodel(
            p, data_dir, True, 300000
        )  # , ckpt=True, ckpt_it=ck_it)

        res = Out_sample_misspec_test(
            p["N_test"],
            None,
            p["factor_lb"],
            p["Startholding"],
            p["CostMultiplier"],
            p["kappa"],
            p["discount_rate"],
            p["executeDRL"],
            False,
            p["executeMV"],
            p["RT"],
            p["KLM"],
            p["executeGP"],
            model,
            0,
            p["recurrent_env"],
            p["unfolding"],
            datatype=p["datatype"],
            mean_process=p["mean_process"],
            lags_mean_process=p["lags_mean_process"],
            vol_process=p["vol_process"],
            distr_noise=p["distr_noise"],
            seed=seed,
            seed_param=p["seedparam"],
            sigmaf=p["sigmaf"],
            f0=p["f0"],
            f_param=p["f_param"],
            sigma=p["sigma"],
            HalfLife=p["HalfLife"],
            uncorrelated=p["uncorrelated"],
            degrees=p["degrees"],
            rng=rng,
        )

        plot_portfolio(res, "DQN", ax2)

        # STUD fullfit
        if htype == "cost":
            experiment = "datatype_t_stud_seed_6"
        elif htype == "risk":
            experiment = "datatype_t_stud_seed_400"

        data_dir = "outputs/DQN/20210108_GPSTUD_df8_final_datatype_t_stud/{}/{}".format(
            length, experiment
        )
        filenamep = os.path.join(data_dir, "config_{}.yaml".format(length))
        p = readConfigYaml(filenamep)

        if "seed_init" not in p:
            p["seed_init"] = p["seed"]

        p["N_test"] = ntest
        rng = np.random.RandomState(p["seed"])
        model, actions = load_DQNmodel(
            p, data_dir, True, 300000
        )  # , ckpt=True, ckpt_it=ck_it)

        res = Out_sample_misspec_test(
            p["N_test"],
            None,
            p["factor_lb"],
            p["Startholding"],
            p["CostMultiplier"],
            p["kappa"],
            p["discount_rate"],
            p["executeDRL"],
            False,
            p["executeMV"],
            p["RT"],
            p["KLM"],
            p["executeGP"],
            model,
            0,
            p["recurrent_env"],
            p["unfolding"],
            datatype=p["datatype"],
            mean_process=p["mean_process"],
            lags_mean_process=p["lags_mean_process"],
            vol_process=p["vol_process"],
            distr_noise=p["distr_noise"],
            seed=seed,
            seed_param=p["seedparam"],
            sigmaf=p["sigmaf"],
            f0=p["f0"],
            f_param=p["f_param"],
            sigma=p["sigma"],
            HalfLife=p["HalfLife"],
            uncorrelated=p["uncorrelated"],
            degrees=p["degrees"],
            rng=rng,
        )

        plot_portfolio(res, "DQN", ax3)

        # GARCH
        if htype == "cost":
            experiment = "factor_lb_1_seed_400"  # distr_noise_normal_seed_400
        elif htype == "risk":
            experiment = "factor_lb_1_seed_169"

        data_dir = "outputs/DQN/20210105_GARCH_factor_lb_1/{}/{}".format(
            length, experiment
        )
        filenamep = os.path.join(data_dir, "config_{}.yaml".format(length))
        p = readConfigYaml(filenamep)

        if "seed_init" not in p:
            p["seed_init"] = p["seed"]
        # p['factor_lb'] = [100]
        p["N_test"] = ntest
        rng = np.random.RandomState(p["seed"])
        model, actions = load_DQNmodel(
            p, data_dir, True, 300000
        )  # , ckpt=True, ckpt_it=ck_it)

        res = Out_sample_misspec_test(
            p["N_test"],
            None,
            p["factor_lb"],
            p["Startholding"],
            p["CostMultiplier"],
            p["kappa"],
            p["discount_rate"],
            p["executeDRL"],
            False,
            p["executeMV"],
            p["RT"],
            p["KLM"],
            p["executeGP"],
            model,
            0,
            p["recurrent_env"],
            p["unfolding"],
            datatype=p["datatype"],
            mean_process=p["mean_process"],
            lags_mean_process=p["lags_mean_process"],
            vol_process=p["vol_process"],
            distr_noise=p["distr_noise"],
            seed=seed,
            seed_param=p["seedparam"],
            sigmaf=p["sigmaf"],
            f0=p["f0"],
            f_param=p["f_param"],
            sigma=p["sigma"],
            HalfLife=p["HalfLife"],
            uncorrelated=p["uncorrelated"],
            degrees=p["degrees"],
            rng=rng,
        )

        plot_portfolio(res, "DQN", ax4)

        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax1.set_title("Gaussian")
        ax2.set_title("Stud fully inf")
        ax3.set_title("Stud partially inf")
        ax4.set_title("Garch")

        ax2.legend(loc=0, prop={"size": 4})

        for ax in fig.axes:
            ax.tick_params(pad=0.0)
        fig.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.15)

        fig.text(0.5, 0.04, "out-of-sample iterations", ha="center")
        fig.text(0.04, 0.5, "holding", va="center", rotation="vertical")

        plt.savefig(os.path.join("outputs", "figs", "holdings.pdf"), dpi=300)

    elif p["plot_type"] == "rewgauss":

        colors = ["blue", "orange"]
        fig = plt.figure(
            figsize=set_size(width=243.9112, subplots=(2, 1))
        )  # 505.89 243.9112
        gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        axes = fig.axes
        axes_pairs = [axes[: len(axes) // 2], axes[len(axes) // 2 :]]
        outputModel = p["outputModel"]
        out_mode_pairs = [
            outputModel[: len(outputModel) // 2],
            outputModel[len(outputModel) // 2 :],
        ]

        for pair, out_mode in zip(axes_pairs, out_mode_pairs):
            runRewplots(p, pair, out_mode)

        # TITLES
        ax1.set_title("Reward")
        # LEGEND
        ax1.legend(loc=4)
        fig.text(0.5, 0.04, "$\mathregular{T_{in}}$", ha="center")
        fig.text(0.04, 0.5, "% benchmark", va="center", rotation="vertical")
        plt.gcf().subplots_adjust(left=0.13, hspace=0.1)

        plt.setp(ax1.get_xticklabels(), visible=False)

        for ax in fig.axes:
            ax.tick_params(pad=0.05)
        fig.savefig(os.path.join("outputs", "figs", "RewGAUSS_performance.pdf"))
        logging.info("Plot saved successfully...")

    elif p["plot_type"] == "rewstud":

        colors = ["blue", "orange"]
        fig = plt.figure(figsize=set_size(width=243.9112, subplots=(2, 1)))

        gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        axes = fig.axes
        axes_pairs = [axes[: len(axes) // 2], axes[len(axes) // 2 :]]
        outputModel = p["outputModel"]
        out_mode_pairs = [
            outputModel[: len(outputModel) // 2],
            outputModel[len(outputModel) // 2 :],
        ]

        for pair, out_mode in zip(axes_pairs, out_mode_pairs):
            runRewplots(p, pair, out_mode)

        # TITLES
        ax1.set_title("Reward")
        # LEGEND
        legend = ax1.legend(loc=4)
        legend.get_texts()[0].set_text("mfit")
        legend.get_texts()[1].set_text("fullfit")
        fig.text(0.5, 0.04, "$\mathregular{T_{in}}$", ha="center")
        fig.text(0.04, 0.5, "% benchmark", va="center", rotation="vertical")
        plt.gcf().subplots_adjust(left=0.13, hspace=0.1)

        plt.setp(ax1.get_xticklabels(), visible=False)

        for ax in fig.axes:
            ax.tick_params(pad=0.0)

        fig.savefig(os.path.join("outputs", "figs", "RewTSTUD_performance.pdf"))

    elif p["plot_type"] == "rewgarch":

        colors = ["blue", "orange"]
        fig = plt.figure(figsize=set_size(width=243.9112))

        ax1 = fig.add_subplot()
        runRewplots(p, fig.axes, p["outputModel"])

        # TITLES
        ax1.set_title("Reward")
        # LEGEND
        legend = ax1.legend()
        legend.get_texts()[0].set_text("normal")
        legend.get_texts()[1].set_text("student's t")

        fig.text(0.5, 0.04, "$\mathregular{T_{in}}$", ha="center")
        fig.text(0.04, 0.5, r"$\Delta$ benchmark", va="center", rotation="vertical")
        plt.gcf().subplots_adjust(left=0.17, bottom=0.2)

        fig.savefig(os.path.join("outputs", "figs", "RewGARCH_performance.pdf"))
        logging.info("Plot saved successfully...")
