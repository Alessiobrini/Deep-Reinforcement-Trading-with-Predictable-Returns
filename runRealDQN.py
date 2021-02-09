# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:09:28 2020

@author: aless
"""

# delete any variables created in previous run if you are using this script on Spyder
import os

if any("SPYDER" in name for name in os.environ):
    from IPython import get_ipython

    get_ipython().magic("reset -sf")

# 0. importing section initialize logger.--------------------------------------
import logging
import os
import pdb
import  sys
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np

from utils.common import (readConfigYaml, saveConfigYaml, generate_logger, GeneratePathFolder)

from utils.simulation import create_lstm_tensor
from utils.env import (
    MarketEnv,
    RecurrentMarketEnv,
    ActionSpace,
)
from utils.DQN import DQN
from utils.test import Out_sample_real_test, empty_series
from utils.tools import get_action_boundaries, CalculateLaggedSharpeRatio, RunModels


# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
Param = readConfigYaml(os.path.join(os.getcwd(), "config", "paramRealDQN.yaml"))
logging.info("Successfully read config file with parameters...")


def RunRealDQNTraders(Param):
    """Main function which loads the parameters for the real experiments
    and run both training and testing routines

    Parameters
    ----------
    Param: dict
        The dictionary containing the parameters
    """
    # 0. EXTRACT PARAMETERS ----------------------------------------------------------
    epsilon = Param["epsilon"]
    min_eps_pct = Param["min_eps_pct"]
    min_eps = Param["min_eps"]
    gamma = Param["gamma"]
    kappa = Param["kappa"]
    std_rwds = Param["std_rwds"] # TODO probably useless for my case
    DQN_type = Param["DQN_type"]
    recurrent_env = Param["recurrent_env"]
    use_PER = Param["use_PER"]
    PER_e = Param["PER_e"]
    PER_a = Param["PER_a"]
    PER_b = Param["PER_b"]
    PER_b_anneal = Param["PER_b_anneal"]
    final_PER_b = Param["final_PER_b"]
    PER_b_steps = Param["PER_b_steps"]
    PER_a_anneal = Param["PER_a_anneal"]
    final_PER_a = Param["final_PER_a"]
    PER_a_steps = Param["PER_a_steps"]
    sample_type = Param["sample_type"]
    selected_loss = Param["selected_loss"]
    activation = Param["activation"]
    kernel_initializer = Param["kernel_initializer"]
    batch_norm_input = Param["batch_norm_input"]
    batch_norm_hidden = Param["batch_norm_hidden"]
    clipgrad = Param["clipgrad"]
    clipnorm = Param["clipnorm"]
    clipvalue = Param["clipvalue"]
    clipglob_steps = Param["clipglob_steps"]
    optimizer_name = Param["optimizer_name"]
    beta_1 = Param["beta_1"]
    beta_2 = Param["beta_2"]
    eps_opt = Param["eps_opt"]
    hidden_units = Param["hidden_units"]
    batch_size = Param["batch_size"]
    max_exp_pct = Param["max_exp_pct"]
    copy_step = Param["copy_step"]
    update_target = Param["update_target"]
    tau = Param["tau"]
    learning_rate = Param["learning_rate"]
    lr_schedule = Param["lr_schedule"]
    exp_decay_pct = Param["exp_decay_pct"]
    exp_decay_rate = Param["exp_decay_rate"]
    hidden_memory_units = Param["hidden_memory_units"]
    unfolding = Param["unfolding"]
    qts = Param["qts"]
    KLM = Param["KLM"]
    zero_action = Param["zero_action"]
    min_n_actions = Param["min_n_actions"]
    # Data Simulation
    asset_class = Param["asset_class"]
    symbol = Param["symbol"]
    split_pct = Param["split_pct"]
    factor_lb = Param["factor_lb"]
    CostMultiplier = Param["CostMultiplier"]
    discount_rate = Param["discount_rate"]
    Startholding = Param["Startholding"]
    # Experiment and storage
    episodes = Param["episodes"]
    start_train = Param["start_train"]
    training = Param['training']
    seed = Param["seed"]
    out_of_sample_test = Param["out_of_sample_test"]
    executeDRL = Param["executeDRL"]
    executeGP = Param["executeGP"]
    executeMV = Param["executeMV"]
    save_results = Param["save_results"]
    plot_hist = Param["plot_hist"]
    plot_steps_hist = Param["plot_steps_hist"]
    plot_steps = Param["plot_steps"]
    save_model = Param["save_model"]
    save_ckpt_model = Param["save_ckpt_model"]
    use_GPU = Param["use_GPU"]
    outputDir = Param["outputDir"]
    outputClass = Param["outputClass"]
    outputModel = Param["outputModel"]
    varying_pars = Param["varying_pars"]

    # set random number generator
    rng = np.random.RandomState(seed)

    if use_GPU:
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
        tf.config.experimental.set_visible_devices(
            devices=my_devices, device_type="CPU"
        )

    if not recurrent_env:
        Param["unfolding"] = unfolding = 1

    if update_target == "soft":
        if Param["varying_type"] == "random_search":
            copy_step = Param["copy_step"] = 1
        else:
            assert copy_step == 1, "Soft target updates require copy step to be 1"


    # 1. IMPORT REAL DATA --------------------------------------------------------------
    data_folder = 'daily_{}'.format(asset_class)
    start_year = '2000'
    
    asset_df = pd.read_parquet(
        "data/{}/daily_bars_{}_{}-07-01_2020-07-15.parquet.gzip".format(
            data_folder, symbol, start_year
        )
    )
    
    # set date as index
    asset_df.set_index("date", inplace=True)
    # revert the order of the data
    asset_df = asset_df.iloc[::-1]
    # get closing price
    price_series = asset_df["close_p"]
    # calculate predicting factors
    df = CalculateLaggedSharpeRatio(price_series, factor_lb, symbol)
    # get split point
    split_obs = int(round(df.shape[0] * split_pct, -1))
    df_test = df.iloc[split_obs:]
    df = df.iloc[:split_obs]
    
    len_series = Param['len_series'] = df.shape[0]
    
    if episodes > 1:
        arr_tmp = df.values
        df = pd.DataFrame(
            np.tile(arr_tmp, (episodes, 1)),
            columns=df.columns,
            index=np.tile(df.index, episodes),
        ).iloc[: (df.shape[0] * episodes) + unfolding + 1]

    N_train = df.shape[0]
    N_test = df_test.shape[0]
    Param['N_train'] = N_train
    Param['N_test'] = N_test
    # 2. SET UP SOME HYPERPARAMETERS --------------------------------------------------------------

    steps_to_min_eps = int(N_train * min_eps_pct)
    Param["steps_to_min_eps"] = steps_to_min_eps

    Param["eps_decay"] = (epsilon - min_eps) / steps_to_min_eps
    eps_decay = Param["eps_decay"]

    max_experiences = int(N_train * max_exp_pct)
    Param["max_experiences"] = max_experiences

    exp_decay_steps = int(N_train * exp_decay_pct)
    Param["exp_decay_steps"] = exp_decay_steps

    if PER_b_anneal:
        Param["PER_b_steps"] = PER_b_steps = (N_train) 
        Param["PER_b_growth"] = (final_PER_b - PER_b) / PER_b_steps
        PER_b_growth = Param["PER_b_growth"]
    else:
        Param["PER_b_growth"] = 0.0
        PER_b_growth = Param["PER_b_growth"]

    if PER_a_anneal:
        Param["PER_a_steps"] = PER_a_steps = N_train 
        Param["PER_a_growth"] = (final_PER_a - PER_a) / PER_a_steps
        PER_a_growth = Param["PER_a_growth"]
    else:
        Param["PER_a_growth"] = 0.0
        PER_a_growth = Param["PER_a_growth"]

    if save_ckpt_model:
        save_ckpt_steps = int(N_train  / save_ckpt_model)
        Param["save_ckpt_steps"] = save_ckpt_steps
    


    # 2. FIT REGRESSIONS --------------------------------------------------------------    
    y, X = df[df.columns[0]], df[df.columns[1:]]
    params_retmodel, params_meanrev, fitted_retmodel, fitted_ous = RunModels(y, X)
    # get results
    dates = df.index
    returns = df.iloc[:, 0].values
    factors = df.iloc[:, 1:].values
    sigma = df.iloc[:, 0].std()
    f_param = params_retmodel["params"]
    f_speed = np.abs(np.array([*params_meanrev.values()]).ravel())
    HalfLife = np.around(np.log(2) / f_speed, 2)

    Param["HalfLife"] = HalfLife
    Param["f_speed"] = f_speed
    Param["f_param"] = f_param
    Param["sigma"] = sigma
    sigmaf = X.std().values 
    Param["sigmaf"] = sigmaf


    if recurrent_env:
        returns_tens = create_lstm_tensor(returns.reshape(-1, 1), unfolding)
        factors_tens = create_lstm_tensor(factors, unfolding)
    logging.info("Successfully loaded data...")

    # 2. PATH FOR MODEL (CKPT) AND TB OUTPUT, STORE CONFIG ---------------
    savedpath = GeneratePathFolder(
        outputDir, outputClass, outputModel, varying_pars, N_train, Param
    )
    saveConfigYaml(Param, savedpath)
    log_dir = os.path.join(savedpath, "tb")
    summary_writer = tf.summary.create_file_writer(log_dir)
    if save_ckpt_model and not os.path.exists(os.path.join(savedpath, "ckpt")):
        os.makedirs(os.path.join(savedpath, "ckpt"))
    elif save_ckpt_model and os.path.exists(os.path.join(savedpath, "ckpt")):
        pass
    logging.info("Successfully generated path and stored config...")

    # 3. CREATE MARKET ENVIRONMENTS --------------------------------------------------------------
    # market env for DQN or its variant

    if recurrent_env:
        env = RecurrentMarketEnv(
            HalfLife,
            Startholding,
            sigma,
            CostMultiplier,
            kappa,
            N_train,
            discount_rate,
            f_param,
            f_speed,
            returns,
            factors,
            returns_tens,
            factors_tens,
            dates=dates,
        )

    else:
        env = MarketEnv(
            HalfLife,
            Startholding,
            sigma,
            CostMultiplier,
            kappa,
            N_train,
            discount_rate,
            f_param,
            f_speed,
            returns,
            factors,
            dates=dates,
        )

    action_quantiles, _, _ = get_action_boundaries(
        HalfLife,
        Startholding,
        sigma,
        CostMultiplier,
        kappa,
        N_train,
        discount_rate,
        f_param,
        f_speed,
        returns[:len_series],
        factors[:len_series],
        qts=qts,
        min_n_actions=min_n_actions
    )
    
    KLM[:2] = action_quantiles
    Param["KLM"] = KLM
    action_space = ActionSpace(KLM, zero_action)
    # market env for tab Q learning
    logging.info("Successfully initialized the market environment...")

    # 4. CREATE INITIAL STATE AND NETWORKS ----------------------------------------------------------
    # instantiate the initial state (return, holding) for DQN
    CurrState, CurrFactors = env.reset()
    # instantiate the initial state for the benchmark
    if executeGP:
        CurrOptState = env.opt_reset()
        OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
    # instantiate the initial state for the markovitz solution
    if executeMV:
        CurrMVState = env.opt_reset()

    input_shape = CurrState.shape

    # create train and target network
    TrainQNet = DQN(
        seed,
        DQN_type,
        recurrent_env,
        gamma,
        max_experiences,
        update_target,
        tau,
        input_shape,
        hidden_units,
        hidden_memory_units,
        batch_size,
        selected_loss,
        learning_rate,
        start_train,
        optimizer_name,
        batch_norm_input,
        batch_norm_hidden,
        activation,
        kernel_initializer,
        plot_hist,
        plot_steps_hist,
        plot_steps,
        summary_writer,
        action_space,
        use_PER,
        PER_e,
        PER_a,
        PER_b,
        final_PER_b,
        PER_b_steps,
        PER_b_growth,
        final_PER_a,
        PER_a_steps,
        PER_a_growth,
        sample_type,
        clipgrad,
        clipnorm,
        clipvalue,
        clipglob_steps,
        beta_1,
        beta_2,
        eps_opt,
        std_rwds,
        lr_schedule,
        exp_decay_steps,
        exp_decay_rate,
        rng=rng,
        modelname="TrainQNet",
    )
    TargetQNet = DQN(
        seed,
        DQN_type,
        recurrent_env,
        gamma,
        max_experiences,
        update_target,
        tau,
        input_shape,
        hidden_units,
        hidden_memory_units,
        batch_size,
        selected_loss,
        learning_rate,
        start_train,
        optimizer_name,
        batch_norm_input,
        batch_norm_hidden,
        activation,
        kernel_initializer,
        plot_hist,
        plot_steps_hist,
        plot_steps,
        summary_writer,
        action_space,
        use_PER,
        PER_e,
        PER_a,
        PER_b,
        final_PER_b,
        PER_b_steps,
        PER_b_growth,
        final_PER_a,
        PER_a_steps,
        PER_a_growth,
        sample_type,
        clipgrad,
        clipnorm,
        clipvalue,
        clipglob_steps,
        beta_1,
        beta_2,
        eps_opt,
        std_rwds,
        lr_schedule,
        exp_decay_steps,
        exp_decay_rate,
        rng=rng,
        modelname="TargetQNet",
    )

    logging.info(
        "Successfully initialized the Deep Q Networks...YOU ARE CURRENTLY USING A SEED TO INITIALIZE WEIGHTS."
    )


    # 5. TRAIN ALGORITHM ----------------------------------------------------------
    
    if training == 'online':
        iterations = [ str(i) for i in np.arange(start=0,stop=N_train,step=save_ckpt_steps)[1:]]
        out_series = empty_series(iterations)
    
        iters = 0
        cycle_len = N_train - 1
    
        for i in tqdm(iterable=range(cycle_len), desc="Training DQNetwork"):
            if executeDRL:
                epsilon = max(min_eps, epsilon - eps_decay)
                shares_traded = TrainQNet.eps_greedy_action(CurrState, epsilon)

                NextState, Result, NextFactors = env.step(CurrState, shares_traded, i)
                env.store_results(Result, i)
    
                exp = {
                    "s": CurrState,
                    "a": shares_traded,
                    "r": Result["Reward_DQN"],
                    "s2": NextState,
                    "f": NextFactors,
                }
                TrainQNet.add_experience(exp)
                TrainQNet.train(TargetQNet, i)
    
                CurrState = NextState
                CurrFactors = NextFactors # TODO understand if it is needed
                iters += 1
                if (iters % copy_step == 0) and (i > TrainQNet.start_train):
                    TargetQNet.copy_weights(TrainQNet)
    
                if (
                    save_ckpt_model
                    and (i % save_ckpt_steps == 0)
                    and (i > TrainQNet.start_train)
                ):
                    TrainQNet.model.save_weights(
                        os.path.join(savedpath, "ckpt", "DQN_{}_it_weights".format(i)),
                        save_format="tf",
                    )
    
            if executeGP:
                NextOptState, OptResult = env.opt_step(
                    CurrOptState, OptRate, DiscFactorLoads, i
                )
                env.store_results(OptResult, i)
                CurrOptState = NextOptState
    
            if executeMV:
                NextMVState, MVResult = env.mv_step(CurrMVState, i)
                env.store_results(MVResult, i)
                CurrMVState = NextMVState
    
            # 5.1 OUT OF SAMPLE TEST ----------------------------------------------------------
            if out_of_sample_test:
                if (i % save_ckpt_steps == 0) and (i != 0) and (i > TrainQNet.start_train):
                    (
                        pnl,
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
                        ) = Out_sample_real_test(
                        N_test,
                        df_test,
                        factor_lb,
                        Startholding,
                        CostMultiplier,
                        kappa,
                        discount_rate,
                        executeDRL,
                        executeMV,
                        KLM,
                        executeGP,
                        TrainQNet,
                        savedpath,
                        i,
                        recurrent_env,
                        unfolding,
                    )
                            
                            
                    out_series.collect(pnl,
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
                                        i)
    elif training == 'offline':
        iterations = [ str(i+1) for i in range(episodes)]
        out_series = empty_series(iterations)
        iters = 0
        for e in tqdm(iterable=range(episodes), desc='Running episodes...'):

            for i in range(len_series - 1):

                epsilon = max(min_eps, epsilon - eps_decay)

                shares_traded = TrainQNet.eps_greedy_action(CurrState, epsilon)

                NextState, Result, NextFactors = env.step(CurrState, shares_traded, i)

                exp = {
                    "s": CurrState,
                    "a": shares_traded,
                    "r": Result["Reward_DQN"],
                    "s2": NextState,
                    "f": NextFactors,
                }
                
                TrainQNet.add_experience(exp)
                CurrState = NextState
                CurrFactors = NextFactors # TODO same as above
                    
            for i in range(len_series - 1):

                TrainQNet.train(TargetQNet, i)
    
                iters += 1
                if (iters % copy_step == 0) and (i > TrainQNet.start_train):
                    TargetQNet.copy_weights(TrainQNet)
    
                if (
                    save_ckpt_model
                    and (i % save_ckpt_steps == 0)
                    and (i > TrainQNet.start_train)
                ):
                    TrainQNet.model.save_weights(
                        os.path.join(savedpath, "ckpt", "DQN_{}_it_weights".format(i)),
                        save_format="tf",
                    )
        
            # 5.1 OUT OF SAMPLE TEST ----------------------------------------------------------
            if out_of_sample_test:
                # if (i % save_ckpt_steps == 0) and (i != 0) and (i > TrainQNet.start_train):
                # pdb.set_trace()
                if (i != 0) and (i > TrainQNet.start_train):
                    
                    (
                        pnl,
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
                        ) = Out_sample_real_test(
                        N_test,
                        df_test,
                        factor_lb,
                        Startholding,
                        CostMultiplier,
                        kappa,
                        discount_rate,
                        executeDRL,
                        executeMV,
                        KLM,
                        executeGP,
                        TrainQNet,
                        savedpath,
                        i,
                        recurrent_env,
                        unfolding,
                    )
                            
                    # pdb.set_trace()
                    out_series.collect(pnl,
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
                                        e+1)

    logging.info("Successfully trained the Deep Q Network...")
    # 6. STORE RESULTS ----------------------------------------------------------
    saveConfigYaml(Param, savedpath)
    if save_results:
        env.save_outputs(savedpath, include_dates=True)
        logging.info("Successfully plotted and stored results...")

    if save_model:
        TrainQNet.model.save_weights(
            os.path.join(savedpath, "DQN_final_weights"), save_format="tf"
        )
        logging.info("Successfully saved DQN weights...")
    if out_of_sample_test:
        out_series.save(savedpath,'DQN', N_test)
        
if __name__ == "__main__":
    RunRealDQNTraders(Param)
