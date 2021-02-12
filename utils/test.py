# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:21 2020

@author: aless
"""
from tqdm import tqdm
from utils.simulation import ReturnSampler, GARCHSampler
from utils.env import MarketEnv, RecurrentMarketEnv, ReturnSpace, HoldingSpace, ActionSpace
from utils.simulation import create_lstm_tensor
from utils.tools import CalculateLaggedSharpeRatio, RunModels
from utils.common import format_tousands
import os
import numpy as np
import pandas as pd
from typing import Union, Optional
from pathlib import Path
import pdb, sys
from utils.tools import get_bet_size


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
    RT: list,
    KLM: list,
    executeGP: bool,
    TrainNet,
    iteration: int,
    savedpath: Union[str or Path] = None,
    recurrent_env: bool = False,
    unfolding: int = 1,
    QTable: Optional[pd.DataFrame] = None,
    rng: int = None,
    seed_test: int = None,
    action_limit=None,
    uncorrelated=False,
    t_stud: bool = False,
    variables: list = None,
    side_only: bool = False,
    discretization: float = None,
    temp: float = 200.0,
    zero_action: bool = True,
    store_values : bool = True,
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
        
    side_only: bool
        Regulate the decoupling between side and size of the bet
        
    discretization: float
        Level of discretization. If none, no discretization will be applied
        
    temp: float
        Temperature of boltzmann equation

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

    if "DQN" in tag:
        action_space = ActionSpace(KLM, zero_action=zero_action, side_only=side_only) # TODO hard coded zero action
    if executeDRL:
        CurrState, _ = test_env.reset()
    if executeRL:
        test_env.returns_space = ReturnSpace(RT)
        test_env.holding_space = HoldingSpace(KLM)
        DiscrCurrState = test_env.discrete_reset()
    if executeGP:
        CurrOptState = test_env.opt_reset()
        OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()


    for i in tqdm(iterable=range(N_test + 1), desc="Testing DQNetwork"):
        if executeDRL:
            if "DQN" in tag:

                qvalues = TrainNet( np.atleast_2d(CurrState.astype("float32")), training=False)
                shares_traded = action_space.values[np.argmax(qvalues[0])]
                
                if side_only:
                    shares_traded = get_bet_size(qvalues,shares_traded,action_limit=KLM[0], rng=rng, 
                                                 zero_action = zero_action,
                                                 discretization=discretization,
                                                 temp=temp)
                
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i
                )
                test_env.store_results(Result, i)
            elif "DDPG" in tag:
                tg = "DDPG"
                shares_traded = TrainNet(
                    np.atleast_2d(CurrState.astype("float32")), training=False
                )
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i, tag=tg
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


    if store_values:
        p_avg, r_avg, sr_avg, absp_avg, absr_avg, abssr_avg, abssr_hold = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
    
        for t in tag:
            # select interesting variables and express as a percentage of the GP results
            pnl_str = list(filter(lambda x: "NetPNL_{}".format(t) in x, variables))
            opt_pnl_str = list(filter(lambda x: "OptNetPNL" in x, variables))
            rew_str = list(filter(lambda x: "Reward_{}".format(t) in x, variables))
            opt_rew_str = list(filter(lambda x: "OptReward" in x, variables))
    
            # pnl
            pnl = test_env.res_df[pnl_str + opt_pnl_str].iloc[:-1]
            cum_pnl = pnl.cumsum()
            ref_pnl = (np.array(cum_pnl[pnl_str]) / np.array(cum_pnl[opt_pnl_str])) * 100
            # rewards
            rew = test_env.res_df[rew_str + opt_rew_str].iloc[:-1]
            cum_rew = rew.cumsum()
            ref_rew = (np.array(cum_rew[rew_str]) / np.array(cum_rew[opt_rew_str])) * 100
    
            # SR
            mean = np.array(pnl[pnl_str]).mean()
            std = np.array(pnl[pnl_str]).std()
            sr = (mean / std) * (252 ** 0.5)
    
            # Holding
            hold = test_env.res_df["NextHolding_{}".format(t)].iloc[
                -2
            ]  # avoid last observation
            opthold = test_env.res_df["OptNextHolding"].iloc[-2]
    
            opt_mean = np.array(pnl[opt_pnl_str]).mean()
            opt_std = np.array(pnl[opt_pnl_str]).std()
            optsr = (opt_mean / opt_std) * (252 ** 0.5)
    
            perc_SR = (sr / optsr) * 100
    
            p_avg.append(ref_pnl[-1])
            r_avg.append(ref_rew[-1])
            sr_avg.append(perc_SR)
    
            absp_avg.append(cum_pnl.iloc[-1].values[0])
            absr_avg.append(cum_rew.iloc[-1].values[0])
            abssr_avg.append(sr)
    
            abssr_hold.append(hold)
    
        # return only the last value of the series which is the cumulated pnl expressed as a percentage of GP
        return (
            np.array(p_avg).ravel(),
            np.array(r_avg).ravel(),
            np.array(sr_avg).ravel(),
            np.array(absp_avg).ravel(),
            cum_pnl.iloc[-1].values[1],
            np.array(absr_avg).ravel(),
            cum_rew.iloc[-1].values[1],
            np.array(abssr_avg).ravel(),
            optsr,
            np.array(abssr_hold).ravel(),
            opthold,
        )
    else:
        if savedpath:
            test_env.save_outputs(savedpath, test=True, iteration=iteration)
        return test_env.res_df


    


def Out_sample_Misspec_test(
    N_test: int,
    df: np.ndarray,
    factor_lb: list,
    Startholding: Union[float or int],
    CostMultiplier: float,
    kappa: float,
    discount_rate: float,
    executeDRL: bool,
    executeRL: bool,
    RT: list,
    KLM: list,
    executeGP: bool,
    TrainNet,
    iteration: int,
    savedpath: Union[str or Path] = None,
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
    variables: list = None,
    side_only: bool = False,
    discretization: float = None,
    temp: float = 200.0,
    zero_action: bool = True,
    tag="DQN",
    store_values : bool = True,
):
    """
    Perform an out-of-sample test and store results in the case of misspecified
    model dynamic

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
    
    side_only: bool
        Regulate the decoupling between side and size of the bet
        
    discretization: float
        Level of discretization. If none, no discretization will be applied
        
    temp: float
        Temperature of boltzmann equation

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
        if factor_lb:
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
        else:
            returns, factors, test_f_speed = ReturnSampler(
                N_test,
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
            )
            y, X = df[df.columns[0]], df[df.columns[1:]]
        dates = df.index
        
    elif datatype == "garch_mr":

        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function

        returns, factors, f_speed = ReturnSampler(
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
            t_stud=False,
            vol = 'heterosk',
        )
        
        df = CalculateLaggedSharpeRatio(
            returns, factor_lb, nameTag=datatype, seriestype="return"
        )
        y, X = df[df.columns[0]], df[df.columns[1:]]
        dates = df.index
    else:
        print("Datatype not correct")
        sys.exit()

    if datatype == "t_stud_mfit":
        params_meanrev, _ = RunModels(y, X, mr_only=True)
    else:
        # do regressions
        params_retmodel, params_meanrev, _, _ = RunModels(y, X)
        # get results
    test_returns = df.iloc[:, 0].values
    test_factors = df.iloc[:, 1:].values

    if datatype != "t_stud_mfit":
        sigma_fit = df.iloc[:, 0].std()
        f_param_fit = params_retmodel["params"]
    else:
        sigma_fit = sigma
        f_param_fit = f_param
    test_f_speed_fit = np.abs(
        np.array([*params_meanrev.values()]).ravel()
    )  # TODO check if abs is correct
    HalfLife_fit = np.around(np.log(2) / test_f_speed_fit, 2)
    # print(sigma,sigma_fit)
    # print(HalfLife,HalfLife_fit)
    # print(test_f_speed,test_f_speed_fit)
    # print(f_param,f_param_fit)
    # pdb.set_trace()
    if recurrent_env:
        test_returns_tens = create_lstm_tensor(test_returns.reshape(-1, 1), unfolding)
        test_factors_tens = create_lstm_tensor(test_factors, unfolding)
        test_env = RecurrentMarketEnv(
            HalfLife_fit,
            Startholding,
            sigma_fit,
            CostMultiplier,
            kappa,
            N_test,
            discount_rate,
            f_param_fit,
            test_f_speed_fit,
            test_returns,
            test_factors,
            test_returns_tens,
            test_factors_tens,
            action_limit,
            dates=dates,
        )
    else:
        test_env = MarketEnv(
            HalfLife_fit,
            Startholding,
            sigma_fit,
            CostMultiplier,
            kappa,
            N_test,
            discount_rate,
            f_param_fit,
            test_f_speed_fit,
            test_returns,
            test_factors,
            action_limit,
            dates=dates,
        )
    if "DQN" in tag:
        action_space = ActionSpace(KLM, zero_action=zero_action, side_only=side_only)
    if executeDRL:
        CurrState, _ = test_env.reset()
    if executeRL:
        test_env.returns_space = ReturnSpace(RT)
        test_env.holding_space = HoldingSpace(KLM)
        DiscrCurrState = test_env.discrete_reset()
    if executeGP:
        CurrOptState = test_env.opt_reset()
        OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()


    if recurrent_env:
        cycle_len = N_test + 1 - (unfolding - 1)
    else:
        cycle_len = N_test + 1

    for i in tqdm(iterable=range(cycle_len), desc="Testing DQNetwork"):
        if executeDRL:
            if tag == "DQN":
                qvalues = TrainNet(np.atleast_2d(CurrState.astype("float32")), training=False)
                shares_traded = action_space.values[np.argmax(qvalues[0])]

                if side_only:
                    shares_traded = get_bet_size(qvalues,shares_traded,action_limit=KLM[0], rng=rng, zero_action=zero_action,
                                                 discretization=discretization,
                                                 temp=temp)
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i
                )
                test_env.store_results(Result, i)
            elif tag == "DDPG":
                tg = "DDPG"
                shares_traded = TrainNet(
                    np.atleast_2d(CurrState.astype("float32")), training=False
                )
                NextState, Result, NextFactors = test_env.step(
                    CurrState, shares_traded, i, tag=tg
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


    if store_values:
        p_avg, r_avg, sr_avg, absp_avg, absr_avg, abssr_avg, abssr_hold = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for t in tag:
            # select interesting variables and express as a percentage of the GP results
            pnl_str = list(filter(lambda x: "NetPNL_{}".format(t) in x, variables))
            opt_pnl_str = list(filter(lambda x: "OptNetPNL" in x, variables))
            rew_str = list(filter(lambda x: "Reward_{}".format(t) in x, variables))
            opt_rew_str = list(filter(lambda x: "OptReward" in x, variables))
    
            # pnl
            pnl = test_env.res_df[pnl_str + opt_pnl_str].iloc[:-1]
            cum_pnl = pnl.cumsum()
    
            if datatype == "garch" or datatype=='garch_mr':
                ref_pnl = np.array(cum_pnl[pnl_str]) - np.array(cum_pnl[opt_pnl_str])
            else:
                ref_pnl = (
                    np.array(cum_pnl[pnl_str]) / np.array(cum_pnl[opt_pnl_str])
                ) * 100
    
            # rewards
            rew = test_env.res_df[rew_str + opt_rew_str].iloc[:-1]
            cum_rew = rew.cumsum()
            if datatype == "garch" or datatype == "garch_mr":
                ref_rew = np.array(cum_rew[rew_str]) - np.array(cum_rew[opt_rew_str])
            else:
                ref_rew = (
                    np.array(cum_rew[rew_str]) / np.array(cum_rew[opt_rew_str])
                ) * 100
    
            # SR
            mean = np.array(pnl[pnl_str]).mean()
            std = np.array(pnl[pnl_str]).std()
            sr = (mean / std) * (252 ** 0.5)
    
            # Holding
            hold = test_env.res_df["NextHolding_{}".format(t)].iloc[
                -2
            ]  # avoid last observation
            opthold = test_env.res_df["OptNextHolding"].iloc[-2]
    
            opt_mean = np.array(pnl[opt_pnl_str]).mean()
            opt_std = np.array(pnl[opt_pnl_str]).std()
            optsr = (opt_mean / opt_std) * (252 ** 0.5)
    
            perc_SR = (sr / optsr) * 100
    
            p_avg.append(ref_pnl[-1])
            r_avg.append(ref_rew[-1])
            sr_avg.append(perc_SR)
    
            absp_avg.append(cum_pnl.iloc[-1].values[0])
            absr_avg.append(cum_rew.iloc[-1].values[0])
            abssr_avg.append(sr)
    
            abssr_hold.append(hold)
    
        # return only the last value of the series which is the cumulated pnl expressed as a percentage of GP
        return (
            np.array(p_avg).ravel(),
            np.array(r_avg).ravel(),
            np.array(sr_avg).ravel(),
            np.array(absp_avg).ravel(),
            cum_pnl.iloc[-1].values[1],
            np.array(absr_avg).ravel(),
            cum_rew.iloc[-1].values[1],
            np.array(abssr_avg).ravel(),
            optsr,
            np.array(abssr_hold).ravel(),
            opthold,
        )
    else:
        if savedpath:
            test_env.save_outputs(savedpath, test=True, iteration=iteration)
        return test_env.res_df

def Out_sample_real_test(
    N_test: int,
    df: np.ndarray,
    factor_lb: list,
    Startholding: Union[float or int],
    CostMultiplier: float,
    kappa: float,
    discount_rate: float,
    executeDRL: bool,
    KLM: list,
    executeGP: bool,
    TrainNet,
    savedpath: Union[str or Path],
    iteration: int,
    recurrent_env: bool = False,
    unfolding: int = 1,
    action_limit=None,
    sigmaf: Union[float or list or np.ndarray] = None,
    f0: Union[float or list or np.ndarray] = None,
    f_param: Union[float or list or np.ndarray] = None,
    sigma: Union[float or list or np.ndarray] = None,
    HalfLife: Union[int or list or np.ndarray] = None,
    side_only: bool = False,
    discretization: float = None,
    temp: float = 200.0,
    zero_action: bool = True,
    tag="DQN",
):


    
    y, X = df[df.columns[0]], df[df.columns[1:]]
    dates = df.index

    # TODO parameters are fitted for the GP solution. Do we want to use parmaeter fitted in the training set?
    params_retmodel, params_meanrev, _, _ = RunModels(y, X)
    test_returns = df.iloc[:, 0].values
    test_factors = df.iloc[:, 1:].values
    sigma = df.iloc[:, 0].std()
    f_param = params_retmodel["params"]
    f_speed = np.abs(np.array([*params_meanrev.values()]).ravel())
    HalfLife = np.around(np.log(2) / f_speed, 2)


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
            f_speed,
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
            f_speed,
            test_returns,
            test_factors,
            action_limit,
            dates=dates,
        )

    if executeDRL:
        CurrState, _ = test_env.reset()
    if executeGP:
        CurrOptState = test_env.opt_reset()
        OptRate, DiscFactorLoads = test_env.opt_trading_rate_disc_loads()



    cycle_len = N_test - 1
    for i in tqdm(iterable=range(cycle_len), desc="Testing DQNetwork"):
        if executeDRL:
            if tag == "DQN":
                # shares_traded = TrainNet.greedy_action(CurrState)
                shares_traded, qvalues = TrainNet.greedy_action(CurrState, side_only=side_only)

                if side_only:
                    shares_traded = get_bet_size(qvalues,shares_traded,action_limit=KLM[0], rng=None, zero_action=zero_action,
                                                 discretization=discretization,
                                                 temp=temp)
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

        if executeGP:
            NextOptState, OptResult = test_env.opt_step(
                CurrOptState, OptRate, DiscFactorLoads, i
            )
            test_env.store_results(OptResult, i)
            CurrOptState = NextOptState


    variables = []
    variables.append("NetPNL_{}".format(tag))
    variables.append("Reward_{}".format(tag))
    variables.append("OptNetPNL")
    variables.append("OptReward")

    # select interesting variables and express as a percentage of the GP results
    pnl_str = list(filter(lambda x: "NetPNL_{}".format(tag) in x, variables))
    opt_pnl_str = list(filter(lambda x: "OptNetPNL" in x, variables))
    rew_str = list(filter(lambda x: "Reward_{}".format(tag) in x, variables))
    opt_rew_str = list(filter(lambda x: "OptReward" in x, variables))

    # pnl
    pnl = test_env.res_df[pnl_str + opt_pnl_str].iloc[:-1]
    cum_pnl = pnl.cumsum()
    ref_pnl = np.array(cum_pnl[pnl_str]) - np.array(cum_pnl[opt_pnl_str])


    # rewards
    rew = test_env.res_df[rew_str + opt_rew_str].iloc[:-1]
    cum_rew = rew.cumsum()
    ref_rew = np.array(cum_rew[rew_str]) - np.array(cum_rew[opt_rew_str])

    # SR
    # pnl = test_env.res_df[pnl_str+opt_pnl_str].iloc[:-1]
    mean = np.array(pnl[pnl_str]).mean()
    std = np.array(pnl[pnl_str]).std()
    sr = (mean / std) * (252 ** 0.5)

    # Holding
    hold = test_env.res_df["NextHolding_{}".format(tag)].iloc[
        -2
    ]  # avoid last observation
    opthold = test_env.res_df["OptNextHolding"].iloc[-2]

    opt_mean = np.array(pnl[opt_pnl_str]).mean()
    opt_std = np.array(pnl[opt_pnl_str]).std()
    optsr = (opt_mean / opt_std) * (252 ** 0.5)

    perc_SR = (sr / optsr) * 100

    return (
        ref_pnl[-1][0], # [0] to get the value instead of the numpy array
        ref_rew[-1][0],
        perc_SR,
        cum_pnl.iloc[-1].values[0],
        cum_pnl.iloc[-1].values[1],
        cum_rew.iloc[-1].values[0],
        cum_rew.iloc[-1].values[1],
        sr,
        optsr,
        hold,
        opthold,
    )


class empty_series:
    
    def __init__(self,iterations):
        self.mean_series_pnl = pd.DataFrame(index=range(1), columns=iterations)
        self.mean_series_rew = pd.DataFrame(index=range(1), columns=iterations)
        self.mean_series_sr = pd.DataFrame(index=range(1), columns=iterations)
    
        self.abs_series_pnl_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_pnl_gp = pd.DataFrame(index=range(1), columns=iterations)
        
        self.abs_series_rew_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_rew_gp = pd.DataFrame(index=range(1), columns=iterations)
        
        self.abs_series_sr_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_sr_gp = pd.DataFrame(index=range(1), columns=iterations)
        
        self.abs_series_hold_rl = pd.DataFrame(index=range(1), columns=iterations)
        self.abs_series_hold_gp = pd.DataFrame(index=range(1), columns=iterations)
        
        
    def collect(self,pnl,
                rew,
                sr,
                abs_prl,
                abs_pgp,
                abs_rewrl,
                abs_rewgp,
                abs_srrl,
                abs_srgp,
                abs_hold,
                abs_opthold,
                ckpt_it):
        
        self.mean_series_pnl.loc[0, str(ckpt_it)] = pnl
        self.mean_series_rew.loc[0, str(ckpt_it)] = rew
        self. mean_series_sr.loc[0, str(ckpt_it)] = sr
       
        self.abs_series_pnl_rl.loc[0, str(ckpt_it)] = abs_prl
        self.abs_series_pnl_gp.loc[0, str(ckpt_it)] = abs_pgp
        
        self.abs_series_rew_rl.loc[0, str(ckpt_it)] = abs_rewrl
        self.abs_series_rew_gp.loc[0, str(ckpt_it)] = abs_rewgp
        
        self.abs_series_sr_rl.loc[0, str(ckpt_it)] = abs_srrl
        self.abs_series_sr_gp.loc[0, str(ckpt_it)] = abs_srgp
        
        self.abs_series_hold_rl.loc[0, str(ckpt_it)] = abs_hold
        self.abs_series_hold_gp.loc[0, str(ckpt_it)] = abs_opthold
            
        
    def save(self,exp_path,tag, N_test):
        self.mean_series_pnl.to_parquet(
            os.path.join(
                exp_path,
                "NetPnl_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag
                ),
            ),
            compression="gzip",
        )
        self.mean_series_rew.to_parquet(
            os.path.join(
                exp_path,
                "Reward_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag
                ),
            ),
            compression="gzip",
        )
        self.mean_series_sr.to_parquet(
            os.path.join(
                exp_path,
                "SR_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), tag),
            ),
            compression="gzip",
        )


        self.abs_series_pnl_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsNetPnl_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_pnl_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsNetPnl_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        self.abs_series_rew_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsRew_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_rew_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsRew_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        self.abs_series_sr_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsSR_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_sr_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsSR_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        self.abs_series_hold_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsHold_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag
                ),
            ),
            compression="gzip",
        )
        self.abs_series_hold_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsHold_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )

