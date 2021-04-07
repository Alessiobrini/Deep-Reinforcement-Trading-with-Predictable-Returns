# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:45:22 2019

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
import numpy as np
from utils.common import (
    generate_logger,
    readConfigYaml,
    saveConfigYaml,
    GeneratePathFolder,
)
from utils.simulation import ReturnSampler, create_lstm_tensor
from utils.env import (
    MarketEnv,
    RecurrentMarketEnv,
    ActionSpace,
    ResActionSpace,
    ReturnSpace,
    HoldingSpace,
    CreateQTable,
)
from utils.DQN import DQN
from utils.tools import get_action_boundaries, get_bet_size


# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
Param = readConfigYaml(os.path.join(os.getcwd(), "config", "paramDQN.yaml"))
logging.info("Successfully read config file with parameters...")


def RunDQNTraders(Param):
    """Main function which loads the parameters for the synthetic experiments
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
    action_type= Param['action_type']
    qts = Param["qts"]
    MV_res = Param['MV_res']
    KLM = Param["KLM"]
    zero_action = Param["zero_action"]
    min_n_actions = Param["min_n_actions"]
    side_only = Param["side_only"]
    discretization = Param["discretization"]
    temp = Param["temp"]
    bcm = Param["bcm"]
    bcm_scale = Param["bcm_scale"]
    RT = Param["RT"]
    tablr = Param["tablr"]
    # Data Simulation
    t_stud = Param["t_stud"]
    HalfLife = Param["HalfLife"]
    f0 = Param["f0"]
    f_param = Param["f_param"]
    sigma = Param["sigma"]
    sigmaf = Param["sigmaf"]
    uncorrelated = Param["uncorrelated"]
    CostMultiplier = Param["CostMultiplier"]
    discount_rate = Param["discount_rate"]
    Startholding = Param["Startholding"]
    # Experiment and storage
    episodes = Param["episodes"]
    start_train = Param["start_train"]
    training = Param["training"]
    seed_ret = Param["seed_ret"]
    seed_init = Param["seed_init"]
    N_train = Param["N_train"]
    len_series = Param["len_series"]
    plot_inputs = Param["plot_inputs"]
    executeDRL = Param["executeDRL"]
    executeRL = Param["executeRL"]
    executeGP = Param["executeGP"]
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

    # set random number generator
    rng = np.random.RandomState(seed_ret)

    if use_GPU:
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
        tf.config.experimental.set_visible_devices(
            devices=my_devices, device_type="CPU"
        )

    if seed_init is None:
        seed_init = seed_ret
        Param["seed_init"] = seed_init

    if not recurrent_env:
        Param["unfolding"] = unfolding = 1

    # comment this if you want to update less frequently even when using polyak update ('soft')
    if update_target == "soft":
        assert copy_step == 1, "Soft target updates require copy step to be 1"

    # 1. SIMULATE SYNTHETIC DATA --------------------------------------------------------------

    if training == "online":
        assert (
            N_train == len_series
        ), "Online training requires N_train equal to len_series"
    elif training == "offline":
        N_train = len_series * episodes
    else:
        print("Training mode not correct")
        sys.exit()

    returns, factors, f_speed = ReturnSampler(
        len_series,
        sigmaf,
        f0,
        f_param,
        sigma,
        plot_inputs,
        HalfLife,
        rng,
        offset=unfolding + 1,
        uncorrelated=uncorrelated,
        t_stud=t_stud,
    )

    if recurrent_env:
        returns_tens = create_lstm_tensor(returns.reshape(-1, 1), unfolding)
        factors_tens = create_lstm_tensor(factors, unfolding)
    logging.info("Successfully simulated data...")

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

    if save_ckpt_model:
        save_ckpt_steps = int(N_train / save_ckpt_model)
        Param["save_ckpt_steps"] = save_ckpt_steps

    # 3. PATH FOR MODEL (CKPT) AND TENSORBOARD OUTPUT, STORE CONFIG FILE ---------------
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

    # 4. INSTANTIATE MARKET ENVIRONMENT --------------------------------------------------------------

    if MV_res:
        action_space = ResActionSpace(KLM[0], zero_action)
    else:
        action_quantiles, ret_quantile, holding_quantile = get_action_boundaries(
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
            qts=qts,
            min_n_actions=min_n_actions,
            action_type=action_type,
        )

        KLM[:2] = action_quantiles
        KLM[2] = holding_quantile
        RT[0] = ret_quantile
        Param["RT"] = RT
        Param["KLM"] = KLM
        action_space = ActionSpace(KLM, zero_action, side_only=side_only)

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
            action_limit=KLM[0],
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
            action_limit=KLM[0],
        )

    # market env for tab Q learning
    if executeRL:
        returns_space = ReturnSpace(RT)
        holding_space = HoldingSpace(KLM)
        QTable = CreateQTable(
            returns_space, holding_space, action_space, tablr, gamma, seed_ret
        )
    logging.info("Successfully initialized the market environment...")

    # 4. CREATE INITIAL STATE AND NETWORKS ----------------------------------------------------------
    input_shape = env.get_state_dim()
    # create train and target network
    TrainQNet = DQN(
        seed_init,
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
        seed_init,
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
        rng,
        modelname="TargetQNet",
    )

    logging.info("Successfully initialized the Deep Q Networks...")

    # 5. TRAIN ALGORITHM ----------------------------------------------------------
    if training == "online":
        CurrState, _ = env.reset()
        if executeRL:
            env.returns_space = returns_space
            env.holding_space = holding_space
            DiscrCurrState = env.discrete_reset()
        if executeGP:
            CurrOptState = env.opt_reset()
            OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()

        # iteration count to decide when copying weights for the Target Network
        iters = 0
        for i in tqdm(iterable=range(N_train + 1), desc="Training DQNetwork"):
            if executeDRL:
                epsilon = max(min_eps, epsilon - eps_decay)
                shares_traded, qvalues = TrainQNet.eps_greedy_action(
                    CurrState, epsilon, side_only=side_only
                )

                if not side_only:
                    unscaled_shares_traded = shares_traded
                else:
                    unscaled_shares_traded = get_bet_size(
                        qvalues,
                        shares_traded,
                        action_limit=KLM[0],
                        zero_action=zero_action,
                        rng=rng,
                        discretization=discretization,
                        temp=temp,
                    )
                # print('state {}'.format(CurrState),
                #       'action {}'.format(unscaled_shares_traded),
                #       'q {}'.format(qvalues))

                if MV_res:
                    NextState, Result, _ = env.MV_res_step(CurrState, unscaled_shares_traded, i)
                else:
                    NextState, Result, _ = env.step(CurrState, unscaled_shares_traded, i)
                env.store_results(Result, i)

                exp = {
                    "s": CurrState,
                    "a": shares_traded,
                    "r": Result["Reward_DQN"],
                    "s2": NextState,
                }

                if bcm and side_only:

                    _, OptResult = env.opt_step(
                        CurrOptState, OptRate, DiscFactorLoads, i
                    )

                    exp_bcm = {
                        "unsc_a": unscaled_shares_traded,
                        "opt_a": OptResult["OptNextAction"],
                    }
                    exp = {**exp, **exp_bcm}

                elif bcm and not side_only:

                    _, OptResult = env.opt_step(
                        CurrOptState, OptRate, DiscFactorLoads, i
                    )

                    exp_bcm = {"opt_a": OptResult["OptNextAction"]}
                    exp = {**exp, **exp_bcm}

                TrainQNet.add_experience(exp)
                TrainQNet.train(TargetQNet, i, side_only, bcm, bcm_scale)

                CurrState = NextState

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
                    if executeRL:
                        QTable.save(os.path.join(savedpath, "ckpt"), i)

            if executeRL:
                shares_traded = QTable.chooseAction(DiscrCurrState, epsilon)
                DiscrNextState, Result = env.discrete_step(
                    DiscrCurrState, shares_traded, i
                )
                env.store_results(Result, i)
                QTable.update(DiscrCurrState, DiscrNextState, shares_traded, Result)
                DiscrCurrState = DiscrNextState

            if executeGP:
                NextOptState, OptResult = env.opt_step(
                    CurrOptState, OptRate, DiscFactorLoads, i
                )
                env.store_results(OptResult, i)
                CurrOptState = NextOptState

    elif training == "offline":
        iters = 0
        for e in tqdm(iterable=range(episodes), desc="Running episodes..."):

            CurrState, _ = env.reset()
            if executeGP:
                CurrOptState = env.opt_reset()
                OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()

            for i in range(len_series - 1):

                epsilon = max(min_eps, epsilon - eps_decay)

                shares_traded, qvalues = TrainQNet.eps_greedy_action(
                    CurrState, epsilon, side_only=side_only
                )

                if not side_only:
                    unscaled_shares_traded = shares_traded
                else:
                    unscaled_shares_traded = get_bet_size(
                        qvalues,
                        shares_traded,
                        action_limit=KLM[0],
                        rng=rng,
                        zero_action=zero_action,
                        discretization=discretization,
                        temp=temp,
                    )
                NextState, Result, _ = env.step(CurrState, unscaled_shares_traded, i)
                env.store_results(Result, i)

                exp = {
                    "s": CurrState,
                    "a": shares_traded,
                    "r": Result["Reward_DQN"],
                    "s2": NextState,
                }

                if bcm and side_only:

                    _, OptResult = env.opt_step(
                        CurrOptState, OptRate, DiscFactorLoads, i
                    )

                    exp_bcm = {
                        "unsc_a": unscaled_shares_traded,
                        "opt_a": OptResult["OptNextAction"],
                    }
                    exp = {**exp, **exp_bcm}

                elif bcm and not side_only:

                    _, OptResult = env.opt_step(
                        CurrOptState, OptRate, DiscFactorLoads, i
                    )

                    exp_bcm = {"opt_a": OptResult["OptNextAction"]}
                    exp = {**exp, **exp_bcm}

                TrainQNet.add_experience(exp)

                CurrState = NextState

                if executeGP:
                    NextOptState, OptResult = env.opt_step(
                        CurrOptState, OptRate, DiscFactorLoads, i
                    )
                    env.store_results(OptResult, i)
                    CurrOptState = NextOptState

            for i in range(len_series - 1):
                TrainQNet.train(TargetQNet, i, side_only, bcm, bcm_scale)

                iters += 1
                if (iters % copy_step == 0) and (iters > TrainQNet.start_train):
                    TargetQNet.copy_weights(TrainQNet)

                if (
                    save_ckpt_model
                    and (i % save_ckpt_steps == 0)
                    and (iters > TrainQNet.start_train)
                ):
                    TrainQNet.model.save_weights(
                        os.path.join(savedpath, "ckpt", "DQN_{}_it_weights".format(i)),
                        save_format="tf",
                    )
        logging.info("Successfully trained the Deep Q Network...")
    # 6. STORE RESULTS ----------------------------------------------------------
    Param["iterations"] = [
        str(int(i)) for i in np.arange(0, N_train + 1, save_ckpt_steps)
    ][1:]
    saveConfigYaml(Param, savedpath)
    if save_results:
        env.save_outputs(savedpath)

    if executeRL and save_table:
        QTable.save(savedpath, N_train)
        logging.info("Successfully plotted and stored results...")

    if save_model:
        TrainQNet.model.save_weights(
            os.path.join(savedpath, "DQN_final_weights"), save_format="tf"
        )
        logging.info("Successfully saved DQN weights...")


if __name__ == "__main__":
    RunDQNTraders(Param)
