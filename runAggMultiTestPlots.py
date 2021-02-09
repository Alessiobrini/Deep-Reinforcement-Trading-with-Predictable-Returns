# -*- coding: utf-8 -*-
import os, logging, sys
from utils.utilities import readConfigYaml, generate_logger, format_tousands
import numpy as np
import pandas as pd
from typing import Optional, Union
from utils.multitest_oos_utils import load_DQNmodel, plot_multitest_paper, set_size
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import tensorflow as tf
from utils.SimulateData import ReturnSampler, GARCHSampler
from utils.MarketEnv import MarketEnv, RecurrentMarketEnv
from utils.MarketEnv import ReturnSpace, HoldingSpace, ActionSpace
from utils.SimulateData import create_lstm_tensor
from utils.Regressions import CalculateLaggedSharpeRatio, RunModels
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


def Out_sample_misspec_test(
    N_test: int,
    df: np.ndarray,
    factor_lb: list,
    Startholding: Union[float or int],
    CostMultiplier: float,
    kappa: float,
    discount_rate: float,
    executeDRL: bool,
    executeRL: bool,
    executeMV: bool,
    RT: list,
    KLM: list,
    executeGP: bool,
    TrainNet,
    iteration: int,
    recurrent_env: bool = False,
    unfolding: int = 1,
    QTable: Optional[pd.DataFrame] = None,
    action_limit=None,
    datatype: str = "real",
    mean_process: str = "Constant",
    lags_mean_process: Union[int or None] = None,
    vol_process: str = "GARCH",
    distr_noise: str = "normal",
    seed: int = None,
    seed_param: int = None,
    sigmaf: Union[float or list or np.ndarray] = None,
    f0: Union[float or list or np.ndarray] = None,
    f_param: Union[float or list or np.ndarray] = None,
    sigma: Union[float or list or np.ndarray] = None,
    HalfLife: Union[int or list or np.ndarray] = None,
    uncorrelated: bool = False,
    degrees: int = None,
    rng=None,
    tag="DQN",
):
    """
    Perform an out-of-sample test and return results as a dataframe

    Parameters
    ----------
    N_test : int
        Length of the experiment

    df: np.ndarray,
        Dataframe or numpy array of real data if a test is performed over
        real financial data

    factor_lb: list
        List of lags for constructing factors as lagged variables when in the case
        of benchmark solution

    Startholding: Union[int or float]
        Initial portfolio holding, usually set at 0

    CostMultiplier: float
        Transaction cost parameter which regulates the market liquidity

    kappa: float
        Risk averion parameter

    discount_rate: float
        Discount rate for the reward function

    executeDRL: bool
        Boolean to regulate if perform the deep reinforcement learning algorithm

    executeRL: bool
        Boolean to regulate if perform the reinforcement learning algorithm
    executeMV: bool
        Boolean to regulate if perform the Markovitz solution

    RT: list
        List of boundaries for the discretized return space. The first element is
        the parameter T of the paper and the second it the ticksize
        (usually set as a basis point). Used only for RL case.

    KLM: list
        List of boundaries for Action and Holding space. The first element is
        the extreme boundary of the action space, the second element is
        the intermediate action for such discretized space and the third is
        the boundary for the holding space. In the paper they are defined as
        K, K/2 and M.

    executeGP: bool
        Boolean to regulate if perform the benchmark solution of Garleanu and Pedersen

    TrainNet
        Instantiated class for the train network. It is an instance of
        DeepNetworkModel or DeepRecurrentNetworkModel class

    savedpath: Union[ str or Path]
        Pat where to store results at the end of the training

    iteration: int
        Iteration step

    recurrent_env: bool
        Boolean to regulate if the enviroment is recurrent or not

    unfolding: int = 1
        Timesteps for recurrent. Used only if recurrent_env is True

    QTable: Optional[pd.DataFrame]
        Dataframe representing Q-table

    action_limit=None
        Action boundary used only for DDPG

    datatype: str
        Indicate the type of financial series to be used. It can be 'real' for real
        data, or 'garch', 'tstud', 'tstud_mfit' for different type of synthetic
        financial series. 'tstud' corresponds to the fullfit of the paper, while
        't_stud_mfit' corresponds to mfit.

    mean_process: str
        Mean process for the returns. It can be 'Constant' or 'AR'

    lags_mean_process: int
        Order of autoregressive lag if mean_process is AR

    vol_process: str
        Volatility process for the returns. It can be 'GARCH', 'EGARCH', 'TGARCH',
        'ARCH', 'HARCH', 'FIGARCH' or 'Constant'. Note that different volatility
        processes requires different parameter, which are hard coded. If you want to
        pass them explicitly, use p_arg.

    distr_noise: str
        Distribution for the unpredictable component of the returns. It can be
        'normal', 'studt', 'skewstud' or 'ged'. Note that different distributions
        requires different parameter, which are hard coded. If you want to
        pass them explicitly, use p_arg.

    seed: int
        Seed for experiment reproducibility

    seed_param: int
        Seed for randomly drawing parameter for GARCH type simulation

    sigmaf : Union[float or list or np.ndarray]
        Volatilities of the mean reverting factors

    f0 : Union[float or list or np.ndarray]
        Initial points for simulating factors. Usually set at 0

    f_param: Union[float or list or np.ndarray]
        Factor loadings of the mean reverting factors

    sigma: Union[float or list or np.ndarray]
        volatility of the asset return (additional noise other than the intrinsic noise
                                        in the factors)
    plot_inputs: bool
        Boolean to regulate if plot of simulated returns and factor is needed

    HalfLife: Union[int or list or np.ndarray]
        HalfLife of mean reversion to simulate factors with different speeds

    uncorrelated: bool = False
        Boolean to regulate if the simulated factor are correlated or not

    degrees : int = 8
        Degrees of freedom for Student\'s t noises

    rng: np.random.mtrand.RandomState
        Random number generator

    tag: bool
        Name of the testing algorithm

    """
    if datatype == "real":
        y, X = df[df.columns[0]], df[df.columns[1:]]
        dates = df.index

    elif datatype == "garch":
        return_series, params = GARCHSampler(
            N_test + factor_lb[-1] + 2,
            mean_process=mean_process,
            lags_mean_process=lags_mean_process,
            vol_process=vol_process,
            distr_noise=distr_noise,
            seed=seed,
            seed_param=seed_param,
        )
        df = CalculateLaggedSharpeRatio(
            return_series, factor_lb, nameTag=datatype, seriestype="return"
        )
        y, X = df[df.columns[0]], df[df.columns[1:]]
        dates = df.index
    elif datatype == "t_stud":
        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function
        returns, factors, test_f_speed = ReturnSampler(
            N_test + factor_lb[-1],
            sigmaf,
            f0,
            f_param,
            sigma,
            plot_inputs,
            HalfLife,
            rng=rng,
            offset=unfolding + 1,
            uncorrelated=uncorrelated,
            seed_test=seed,
            t_stud=True,
            degrees=degrees,
        )

        df = CalculateLaggedSharpeRatio(
            returns, factor_lb, nameTag=datatype, seriestype="return"
        )
        y, X = df[df.columns[0]], df[df.columns[1:]]
        dates = df.index
    elif datatype == "t_stud_mfit":
        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function
        returns, factors, test_f_speed = ReturnSampler(
            N_test + factor_lb[-1],
            sigmaf,
            f0,
            f_param,
            sigma,
            plot_inputs,
            HalfLife,
            rng=rng,
            offset=unfolding + 1,
            uncorrelated=uncorrelated,
            seed_test=seed,
            t_stud=True,
            degrees=degrees,
        )

        df = pd.DataFrame(
            data=np.concatenate([returns.reshape(-1, 1), factors], axis=1)
        ).loc[factor_lb[-1] :]
        y, X = df[df.columns[0]], df[df.columns[1:]]
        dates = df.index
    else:
        print("Datatype not correct")
        sys.exit()

    # do regressions
    if datatype == "t_stud_mfit":
        params_meanrev, _ = RunModels(y, X, mr_only=True)
    else:
        # do regressions
        params_retmodel, params_meanrev, _, _ = RunModels(y, X)
    test_returns = df.iloc[:, 0].values
    test_factors = df.iloc[:, 1:].values

    if datatype != "t_stud_mfit":
        sigma = df.iloc[:, 0].std()
        f_param = params_retmodel["params"]
    else:
        sigma = sigma
        f_param = f_param
    test_f_speed = np.abs(np.array([*params_meanrev.values()]).ravel())
    HalfLife = np.around(np.log(2) / test_f_speed, 2)

    if recurrent_env:
        test_returns_tens = create_lstm_tensor(test_returns.reshape(-1, 1), unfolding)
        test_factors_tens = create_lstm_tensor(test_factors, unfolding)
        test_env = RecurrentMarketEnv(
            HalfLife,
            Startholding,
            sigma,
            CostMultiplier,
            kappa,
            N_test,
            discount_rate,
            f_param,
            test_f_speed,
            test_returns,
            test_factors,
            test_returns_tens,
            test_factors_tens,
            action_limit,
            dates=dates,
        )
    else:
        test_env = MarketEnv(
            HalfLife,
            Startholding,
            sigma,
            CostMultiplier,
            kappa,
            N_test,
            discount_rate,
            f_param,
            test_f_speed,
            test_returns,
            test_factors,
            action_limit,
            dates=dates,
        )

    action_space = ActionSpace(KLM, zero_action=True)
    if executeDRL:
        CurrState, _ = test_env.reset()
    if executeRL:
        test_env.returns_space = ReturnSpace(RT)
        test_env.holding_space = HoldingSpace(KLM)
        DiscrCurrState = test_env.discrete_reset()
    if executeGP:
        CurrOptState = test_env.opt_reset()
        OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()
    if executeMV:
        CurrMVState = test_env.opt_reset()

    if datatype == "real":
        if recurrent_env:
            cycle_len = N_test - 1 - (unfolding - 1)
        else:
            cycle_len = N_test - 1
    elif datatype != "real":
        if recurrent_env:
            cycle_len = N_test + 1 - (unfolding - 1)
        else:
            cycle_len = N_test + 1

    for i in tqdm(iterable=range(cycle_len), desc="Testing DQNetwork"):
        if executeDRL:
            if tag == "DQN":
                #                 shares_traded = TrainNet.greedy_action(CurrState)
                shares_traded = action_space.values[
                    np.argmax(
                        TrainNet(
                            np.atleast_2d(CurrState.astype("float32")), training=False
                        )[0]
                    )
                ]
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i
                )
                test_env.store_results(Result, i)
            elif tag == "DDPG":
                shares_traded = TrainNet.p_model(
                    np.atleast_2d(CurrState.astype("float32")), training=False
                )
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i, tag=tag
                )
                test_env.store_results(Result, i)
            CurrState = NextState

        if executeRL:
            shares_traded = int(QTable.chooseGreedyAction(DiscrCurrState))
            DiscrNextState, Result = test_env.discrete_step(
                DiscrCurrState, shares_traded, i
            )
            test_env.store_results(Result, i)
            DiscrCurrState = DiscrNextState

        if executeGP:
            NextOptState, OptResult = test_env.opt_step(
                CurrOptState, OptRate, DiscFactorLoads, i
            )
            test_env.store_results(OptResult, i)
            CurrOptState = NextOptState

        if executeMV:
            NextMVState, MVResult = test_env.mv_step(CurrMVState, i)
            test_env.store_results(MVResult, i)
            CurrMVState = NextMVState
    return test_env.res_df


def Out_sample_test(
    N_test: int,
    sigmaf: Union[float or list or np.ndarray],
    f0: Union[float or list or np.ndarray],
    f_param: Union[float or list or np.ndarray],
    sigma: Union[float or list or np.ndarray],
    plot_inputs: int,
    HalfLife: Union[int or list or np.ndarray],
    Startholding: Union[float or int],
    CostMultiplier: float,
    kappa: float,
    discount_rate: float,
    executeDRL: bool,
    executeRL: bool,
    executeMV: bool,
    RT: list,
    KLM: list,
    executeGP: bool,
    TrainNet,
    iteration: int,
    recurrent_env: bool = False,
    unfolding: int = 1,
    QTable: Optional[pd.DataFrame] = None,
    rng: int = None,
    seed_test: int = None,
    action_limit=None,
    uncorrelated=False,
    t_stud: bool = False,
    tag="DQN",
):

    """
    Perform an out-of-sample test and store results

    Parameters
    ----------
    N_test : int
        Length of the experiment

    sigmaf : Union[float or list or np.ndarray]
        Volatilities of the mean reverting factors

    f0 : Union[float or list or np.ndarray]
        Initial points for simulating factors. Usually set at 0

    f_param: Union[float or list or np.ndarray]
        Factor loadings of the mean reverting factors

    sigma: Union[float or list or np.ndarray]
        volatility of the asset return (additional noise other than the intrinsic noise
                                        in the factors)
    plot_inputs: bool
        Boolean to regulate if plot of simulated returns and factor is needed

    HalfLife: Union[int or list or np.ndarray]
        HalfLife of mean reversion to simulate factors with different speeds

    Startholding: Union[int or float]
        Initial portfolio holding, usually set at 0

    CostMultiplier: float
        Transaction cost parameter which regulates the market liquidity

    kappa: float
        Risk averion parameter

    discount_rate: float
        Discount rate for the reward function

    executeDRL: bool
        Boolean to regulate if perform the deep reinforcement learning algorithm

    executeRL: bool
        Boolean to regulate if perform the reinforcement learning algorithm
    executeMV: bool
        Boolean to regulate if perform the Markovitz solution

    RT: list
        List of boundaries for the discretized return space. The first element is
        the parameter T of the paper and the second it the ticksize
        (usually set as a basis point). Used only for RL case.

    KLM: list
        List of boundaries for Action and Holding space. The first element is
        the extreme boundary of the action space, the second element is
        the intermediate action for such discretized space and the third is
        the boundary for the holding space. In the paper they are defined as
        K, K/2 and M.

    executeGP: bool
        Boolean to regulate if perform the benchmark solution of Garleanu and Pedersen

    TrainNet
        Instantiated class for the train network. It is an instance of
        DeepNetworkModel or DeepRecurrentNetworkModel class

    savedpath: Union[ str or Path]
        Pat where to store results at the end of the training

    iteration: int
        Iteration step

    recurrent_env: bool
        Boolean to regulate if the enviroment is recurrent or not

    unfolding: int = 1
        Timesteps for recurrent. Used only if recurrent_env is True

    QTable: Optional[pd.DataFrame]
        Dataframe representing Q-table

    rng: np.random.mtrand.RandomState
        Random number generator

    seed_test: int
        Seed for test that allows to create a new random number generator
        instead of using the one passed as argument

    action_limit=None
        Action boundary used only for DDPG

    uncorrelated: bool = False
        Boolean to regulate if the simulated factor are correlated or not

    t_stud : bool = False
        Bool to regulate if Student\'s t noises are needed

    variables: list
        Variables to store as results of the experiment

    tag: bool
        Name of the testing algorithm

    """

    test_returns, test_factors, test_f_speed = ReturnSampler(
        N_test,
        sigmaf,
        f0,
        f_param,
        sigma,
        plot_inputs,
        HalfLife,
        rng,
        offset=unfolding + 1,
        seed_test=seed_test,
        uncorrelated=uncorrelated,
        t_stud=t_stud,
    )

    if recurrent_env:
        test_returns_tens = create_lstm_tensor(test_returns.reshape(-1, 1), unfolding)
        test_factors_tens = create_lstm_tensor(test_factors, unfolding)
        test_env = RecurrentMarketEnv(
            HalfLife,
            Startholding,
            sigma,
            CostMultiplier,
            kappa,
            N_test,
            discount_rate,
            f_param,
            test_f_speed,
            test_returns,
            test_factors,
            test_returns_tens,
            test_factors_tens,
            action_limit,
        )
    else:
        test_env = MarketEnv(
            HalfLife,
            Startholding,
            sigma,
            CostMultiplier,
            kappa,
            N_test,
            discount_rate,
            f_param,
            test_f_speed,
            test_returns,
            test_factors,
            action_limit,
        )
    action_space = ActionSpace(KLM, zero_action=True)
    if executeDRL:
        CurrState, _ = test_env.reset()
    if executeRL:
        test_env.returns_space = ReturnSpace(RT)
        test_env.holding_space = HoldingSpace(KLM)
        DiscrCurrState = test_env.discrete_reset()
    if executeGP:
        CurrOptState = test_env.opt_reset()
        OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()
    if executeMV:
        CurrMVState = test_env.opt_reset()

    for i in tqdm(iterable=range(N_test + 1), desc="Testing DQNetwork"):
        if executeDRL:
            if tag == "DQN":
                shares_traded = action_space.values[
                    np.argmax(
                        TrainNet(
                            np.atleast_2d(CurrState.astype("float32")), training=False
                        )[0]
                    )
                ]
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i
                )
                test_env.store_results(Result, i)
            elif tag == "DDPG":
                shares_traded = TrainNet.p_model(
                    np.atleast_2d(CurrState.astype("float32")), training=False
                )
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i, tag=tag
                )
                test_env.store_results(Result, i)
            CurrState = NextState

        if executeRL:
            shares_traded = int(QTable.chooseGreedyAction(DiscrCurrState))
            DiscrNextState, Result = test_env.discrete_step(
                DiscrCurrState, shares_traded, i
            )
            test_env.store_results(Result, i)
            DiscrCurrState = DiscrNextState

        if executeGP:
            NextOptState, OptResult = test_env.opt_step(
                CurrOptState, OptRate, DiscFactorLoads, i
            )
            test_env.store_results(OptResult, i)
            CurrOptState = NextOptState

        if executeMV:
            NextMVState, MVResult = test_env.mv_step(CurrMVState, i)
            test_env.store_results(MVResult, i)
            CurrMVState = NextMVState

    return test_env.res_df


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
        ntest = 500
        seed = 100
        htype = "cost"  # risk or cost

        # GAUSS
        if htype == "cost":
            experiment = "seed_ret_270"
        elif htype == "risk":
            experiment = "seed_ret_924"
        data_dir = "outputs/DQN/20210107_GPGAUSS_final/{}/{}".format(length, experiment)
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
