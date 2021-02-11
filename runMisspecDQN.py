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
import sys
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
from utils.common import (
    generate_logger,
    readConfigYaml,
    saveConfigYaml,
    GeneratePathFolder,
)
from utils.simulation import ReturnSampler, create_lstm_tensor, GARCHSampler
from utils.env import (
    MarketEnv,
    RecurrentMarketEnv,
    ActionSpace,
    ReturnSpace,
    HoldingSpace,
    CreateQTable,
)
from utils.DQN import DQN
from utils.test import Out_sample_Misspec_test
from utils.math_tools import unscale_action
from utils.tools import get_action_boundaries, get_bet_size, CalculateLaggedSharpeRatio, RunModels

# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
Param = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMisspecDQN.yaml"))
logging.info("Successfully read config file with parameters...")


def RunMisspecDQNTraders(Param):
    """Main function which loads the parameters for the synthetic experiments
    and run both training and testing routines

    Parameters
    ----------
    Param: dict
        The dictionary containing the parameters
    """
    # 0. EXTRACT PARAMETERS ----------------------------------------------------------
    plot_inputs = Param["plot_inputs"]
    uncorrelated = Param["uncorrelated"]
    epsilon = Param["epsilon"]
    min_eps_pct = Param["min_eps_pct"]
    min_eps = Param["min_eps"]
    gamma = Param["gamma"]
    kappa = Param["kappa"]
    std_rwds = Param["std_rwds"]
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
    side_only = Param['side_only']
    discretization = Param['discretization']
    temp = Param['temp']
    RT = Param["RT"]
    tablr = Param["tablr"]
    max_exp_pct = Param["max_exp_pct"]
    # Data Simulation
    datatype = Param["datatype"]
    factor_lb = Param["factor_lb"]
    CostMultiplier = Param["CostMultiplier"]
    discount_rate = Param["discount_rate"]
    Startholding = Param["Startholding"]
    # Experiment and storage
    start_train = Param["start_train"]
    seed = Param["seed"]
    seed_param = Param["seed_param"]
    out_of_sample_test = Param["out_of_sample_test"]
    executeDRL = Param["executeDRL"]
    executeRL = Param["executeRL"]
    executeGP = Param["executeGP"]
    executeMV = Param["executeMV"]
    save_results = Param["save_results"]
    save_table = Param["save_table"]
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


    N_train = Param["N_train"]
    N_test = Param["N_test"]
    mean_process = Param["mean_process"]
    lags_mean_process = Param["lags_mean_process"]
    vol_process = Param["vol_process"]
    distr_noise = Param["distr_noise"]
    HalfLife = Param["HalfLife"]
    f0 = Param["f0"]
    f_param = Param["f_param"]
    sigma = Param["sigma"]
    sigmaf = Param["sigmaf"]
    degrees = Param["degrees"]

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

    if PER_b_anneal:
        Param["PER_b_steps"] = PER_b_steps = N_train
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

    # set random number generator
    rng = np.random.RandomState(seed)
    # 1. GET REAL OR SYNTHETIC DATA AND MAKE REGRESSIONS --------------------------------------------------------------
    # import and rearrange data
    if datatype == "garch":
        return_series, params = GARCHSampler(
            N_train + factor_lb[-1] + unfolding + 1,
            mean_process=mean_process,
            lags_mean_process=lags_mean_process,
            vol_process=vol_process,
            distr_noise=distr_noise,
            seed=seed,
            seed_param=seed_param,
        )
        params = {"_".join([datatype, k]): v for k, v in params.items()}
        Param = {**Param, **params}

        df = CalculateLaggedSharpeRatio(
            return_series, factor_lb, nameTag=datatype, seriestype="return"
        )

        y, X = df[df.columns[0]], df[df.columns[1:]]
    elif datatype == "t_stud":
        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function
        returns, factors, f_speed = ReturnSampler(
            N_train + factor_lb[-1],
            sigmaf,
            f0,
            f_param,
            sigma,
            plot_inputs,
            HalfLife,
            rng=rng,
            offset=unfolding + 1,
            uncorrelated=uncorrelated,
            t_stud=True,
            degrees=degrees,
        )
        df = CalculateLaggedSharpeRatio(
            returns, factor_lb, nameTag=datatype, seriestype="return"
        )
        y, X = df[df.columns[0]], df[df.columns[1:]]
    elif datatype == "t_stud_mfit":
        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function
        if factor_lb:
            returns, factors, f_speed = ReturnSampler(
                N_train + factor_lb[-1],
                sigmaf,
                f0,
                f_param,
                sigma,
                plot_inputs,
                HalfLife,
                rng=rng,
                offset=unfolding + 1,
                uncorrelated=uncorrelated,
                t_stud=True,
                degrees=degrees,
            )
            df = pd.DataFrame(
                data=np.concatenate([returns.reshape(-1, 1), factors], axis=1)
            ).loc[factor_lb[-1] :]
            y, X = df[df.columns[0]], df[df.columns[1:]]
        else:
            returns, factors, f_speed = ReturnSampler(
                N_train,
                sigmaf,
                f0,
                f_param,
                sigma,
                plot_inputs,
                HalfLife,
                rng=rng,
                offset=unfolding + 1,
                uncorrelated=uncorrelated,
                t_stud=True,
                degrees=degrees,
            )
            df = pd.DataFrame(
                data=np.concatenate([returns.reshape(-1, 1), factors], axis=1)
            )
            y, X = df[df.columns[0]], df[df.columns[1:]]
            
            
    elif datatype == "garch_mr":
        
        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function

        returns, factors, f_speed = ReturnSampler(
            N_train + factor_lb[-1],
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
        
    else:
        print("Datatype not correct")
        sys.exit()

    # do regressions

    if datatype == "t_stud_mfit":
        params_meanrev, fitted_ous = RunModels(y, X, mr_only=True)
        dates = range(len(returns))
    else:
        params_retmodel, params_meanrev, fitted_retmodel, fitted_ous = RunModels(y, X)
        dates = df.index
    returns = df.iloc[:, 0].values
    factors = df.iloc[:, 1:].values

    # store GP parameters
    if datatype != "t_stud_mfit":
        sigma_fit = df.iloc[:, 0].std()
        f_param_fit = params_retmodel["params"]
        sigmaf_fit = X.std().values
    else:
        sigma_fit = sigma
        f_param_fit = f_param
        sigmaf_fit = sigmaf
    f_speed_fit = np.abs(
        np.array([*params_meanrev.values()]).ravel()
    )  
    HalfLife_fit = np.around(np.log(2) / f_speed_fit, 2)

    if datatype == "garch" or datatype == "real":
        Param["HalfLife"] = HalfLife_fit
        Param["f_speed"] = f_speed_fit
        Param["f_param"] = f_param_fit
        Param["sigma"] = sigma_fit
        Param["sigmaf"] = sigmaf_fit
    elif datatype == "t_stud" or datatype == 'garch_mr':
        Param["HalfLife_fit"] = HalfLife_fit
        Param["f_speed_fit"] = f_speed_fit
        Param["f_param_fit"] = f_param_fit
        Param["sigma_fit"] = sigma_fit
        Param["sigmaf_fit"] = sigmaf_fit
    elif datatype == "t_stud_mfit":
        Param["HalfLife_fit"] = HalfLife_fit
        Param["f_speed_fit"] = f_speed_fit
        Param["f_param_fit"] = f_param
        Param["sigma_fit"] = sigma
        Param["sigmaf_fit"] = sigmaf

    # print(sigma,sigma_fit)
    # print(HalfLife,HalfLife_fit)
    # print(f_speed,f_speed_fit)
    # print(f_param,f_param_fit)
    # pdb.set_trace()


    steps_to_min_eps = int(N_train * min_eps_pct)
    Param["steps_to_min_eps"] = steps_to_min_eps
    eps_decay = (epsilon - min_eps) / steps_to_min_eps
    Param["eps_decay"] = eps_decay

    max_experiences = int(N_train * max_exp_pct)
    Param["max_experiences"] = max_experiences
    exp_decay_steps = int(N_train * exp_decay_pct)
    Param["exp_decay_steps"] = exp_decay_steps

    if save_ckpt_model:
        save_ckpt_steps = int(N_train / save_ckpt_model)

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

    action_quantiles, ret_quantile, holding_quantile = get_action_boundaries(
        HalfLife_fit,
        Startholding,
        sigma_fit,
        CostMultiplier,
        kappa,
        N_train,
        discount_rate,
        f_param_fit,
        f_speed_fit,
        returns,
        factors,
        qts=qts,
        min_n_actions=min_n_actions,
    )

    KLM[:2] = action_quantiles
    KLM[2] = holding_quantile
    RT[0] = ret_quantile
    Param["RT"] = RT
    Param["KLM"] = KLM

    action_space = ActionSpace(KLM, zero_action, side_only=side_only)

    if recurrent_env:
        env = RecurrentMarketEnv(
            HalfLife_fit,
            Startholding,
            sigma_fit,
            CostMultiplier,
            kappa,
            N_train,
            discount_rate,
            f_param_fit,
            f_speed_fit,
            returns,
            factors,
            returns_tens,
            factors_tens,
            action_limit=KLM[0],
            dates=dates,
        )

    else:
        env = MarketEnv(
            HalfLife_fit,
            Startholding,
            sigma_fit,
            CostMultiplier,
            kappa,
            N_train,
            discount_rate,
            f_param_fit,
            f_speed_fit,
            returns,
            factors,
            action_limit=KLM[0],
            dates=dates,
        )


    # market env for tab Q learning
    if executeRL:
        returns_space = ReturnSpace(RT)
        holding_space = HoldingSpace(KLM)
        QTable = CreateQTable(
            returns_space, holding_space, action_space, tablr, gamma, seed
        )
    logging.info("Successfully initialized the market environment...")

    # 4. CREATE INITIAL STATE AND NETWORKS ----------------------------------------------------------
    # instantiate the initial state (return, holding) for DQN
    CurrState, CurrFactors = env.reset()
    # instantiate the initial state (return, holding) for TabQ
    if executeRL:
        env.returns_space = returns_space
        env.holding_space = holding_space
        DiscrCurrState = env.discrete_reset()
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
        "Successfully initialized the Deep Q Networks..."
    )


    # 5. TRAIN ALGORITHM ----------------------------------------------------------
    iters = 0
    if datatype == "real":
        cycle_len = N_train - 1
    elif datatype != "real":
        cycle_len = N_train + 1

    for i in tqdm(iterable=range(cycle_len), desc="Training DQNetwork"):
        if executeDRL:
            epsilon = max(min_eps, epsilon - eps_decay)
            shares_traded, qvalues = TrainQNet.eps_greedy_action(CurrState, epsilon, side_only=side_only)

            if not side_only:
                unscaled_shares_traded = shares_traded
            else:
                unscaled_shares_traded = get_bet_size(qvalues,shares_traded,action_limit=KLM[0], 
                                                      rng=rng,
                                                      discretization=discretization,
                                                      temp=temp)
            # print('state {}'.format(CurrState),
            #       'action {}'.format(unscaled_shares_traded),
            #       'q {}'.format(qvalues))

            NextState, Result, NextFactors = env.step(CurrState, unscaled_shares_traded, i)
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
            CurrFactors = NextFactors # TODO see if it is useful
            iters += 1
            if (iters % copy_step == 0) and (i > TrainQNet.start_train):
                TargetQNet.copy_weights(TrainQNet)

            if (
                save_ckpt_model
                and (i % save_ckpt_steps == 0)
                and (i > TrainQNet.start_train)
            ):
                # if save_ckpt_model and (i % save_ckpt_steps == 0):
                TrainQNet.model.save_weights(
                    os.path.join(savedpath, "ckpt", "DQN_{}_it_weights".format(i)),
                    save_format="tf",
                )
                if executeRL:
                    QTable.save(os.path.join(savedpath, "ckpt"), i)

        if executeRL:
            shares_traded = QTable.chooseAction(DiscrCurrState, epsilon)
            DiscrNextState, Result = env.discrete_step(DiscrCurrState, shares_traded, i)
            env.store_results(Result, i)
            QTable.update(DiscrCurrState, DiscrNextState, shares_traded, Result)
            DiscrCurrState = DiscrNextState

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
                if not executeRL:
                    QTable = None
                Out_sample_Misspec_test(
                    N_test,
                    None,
                    factor_lb,
                    Startholding,
                    CostMultiplier,
                    kappa,
                    discount_rate,
                    executeDRL,
                    executeRL,
                    executeMV,
                    RT,
                    KLM,
                    executeGP,
                    TrainQNet,
                    savedpath,
                    i,
                    recurrent_env,
                    unfolding,
                    QTable,
                    datatype=datatype,
                    mean_process=mean_process,
                    lags_mean_process=lags_mean_process,
                    vol_process=vol_process,
                    distr_noise=distr_noise,
                    seed=seed,
                    seed_param=seed_param,
                    sigmaf=sigmaf,
                    f0=f0,
                    f_param=f_param,
                    sigma=sigma,
                    HalfLife=HalfLife,
                    uncorrelated=uncorrelated,
                    degrees=degrees,
                    rng=rng,
                )


    logging.info("Successfully trained the Deep Q Network...")
    # 6. STORE RESULTS ----------------------------------------------------------
    if not out_of_sample_test:
        Param["iterations"] = [
            str(int(i)) for i in np.arange(0, N_train + 1, save_ckpt_steps)
        ][1:]
    saveConfigYaml(Param, savedpath)
    if save_results:
        env.save_outputs(savedpath, include_dates=True)

    if executeRL and save_table:
        QTable.save(savedpath, N_train)
        logging.info("Successfully plotted and stored results...")

    if save_model:
        TrainQNet.model.save_weights(
            os.path.join(savedpath, "DQN_final_weights"), save_format="tf"
        )
        logging.info("Successfully saved DQN weights...")


if __name__ == "__main__":
    RunMisspecDQNTraders(Param)