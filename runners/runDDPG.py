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


# 0. importing section initialize loggers.--------------------------------------
import logging
import os
import pdb
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils.common import (
    generate_logger,
    GeneratePathFolder,
    readConfigYaml,
    saveConfigYaml,
)
from utils.simulation import ReturnSampler, create_lstm_tensor
from utils.env import (
    MarketEnv,
    RecurrentMarketEnv,
)
from utils.DDPG import DDPG
from utils.test import Out_sample_test
from utils.tools import get_action_boundaries


# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
Param = readConfigYaml(os.path.join(os.getcwd(), "config", "paramDDPG.yaml"))
logging.info("Successfully read config file with parameters...")


def RunDDPGTraders(Param):
    """Main function which loads the parameters for the synthetic experiments
    and run both training and testing routines

    Parameters
    ----------
    Param: dict
        The dictionary containing the parameters
    """
    # 0. EXTRACT PARAMETERS ----------------------------------------------------------
    mu = Param["mu"]
    noise = Param["noise"]
    stddev_noise = Param["stddev_noise"]
    stddev_pol_noise = Param["stddev_pol_noise"]
    noise_decay_pct = Param["noise_decay_pct"]
    theta = Param["theta"]
    gamma = Param["gamma"]
    kappa = Param["kappa"]
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
    selected_loss = Param["selected_loss"]
    activation_Q = Param["activation_Q"]
    activation_p = Param["activation_p"]
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
    hidden_units_Q = Param["hidden_units_Q"]
    hidden_units_p = Param["hidden_units_p"]
    batch_size = Param["batch_size"]
    max_exp_pct = Param["max_exp_pct"]
    copy_step = Param["copy_step"]
    update_target = Param["update_target"]
    tau_Q = Param["tau_Q"]
    tau_p = Param["tau_p"]
    learning_rate_Q = Param["learning_rate_Q"]
    learning_rate_p = Param["learning_rate_p"]
    lr_schedule = Param["lr_schedule"]
    exp_decay_pct = Param["exp_decay_pct"]
    exp_decay_rate_Q = Param["exp_decay_rate_Q"]
    exp_decay_rate_p = Param["exp_decay_rate_p"]
    weight_decay_Q = Param["weight_decay_Q"]
    weight_decay_p = Param["weight_decay_p"]
    qts = Param["qts"]
    KLM = Param["KLM"]
    DDPG_type = Param["DDPG_type"]
    noise_clip = Param["noise_clip"]
    action_limit = Param["action_limit"]
    output_init = Param["output_init"]
    delayed_actions = Param["delayed_actions"]
    recurrent_env = Param["recurrent_env"]
    hidden_memory_units = Param["hidden_memory_units"]
    unfolding = Param["unfolding"]
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
    start_train = Param["start_train"]
    seed_ret = Param["seed_ret"]
    seed_init = Param["seed_init"]
    N_train = Param["N_train"]
    out_of_sample_test = Param["out_of_sample_test"]
    N_test = Param["N_test"]
    plot_inputs = Param["plot_inputs"]
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

    # set random number generator
    rng = np.random.RandomState(seed_ret)

    max_experiences = int(N_train * max_exp_pct)
    Param["max_experiences"] = max_experiences

    exp_decay_steps = int(N_train * exp_decay_pct)
    Param["exp_decay_steps"] = exp_decay_steps

    if noise_decay_pct:
        steps_to_min_stddev_noise = int(N_train * noise_decay_pct)
        Param["steps_to_min_stddev_noise"] = steps_to_min_stddev_noise
        Param["stddev_noise_decay"] = (stddev_noise) / steps_to_min_stddev_noise
        stddev_noise_decay = Param["stddev_noise_decay"]
    else:
        Param["stddev_noise_decay"] = 0.0
        stddev_noise_decay = Param["stddev_noise_decay"]

    if not recurrent_env:
        Param["unfolding"] = unfolding = 1

    if PER_b_anneal:
        Param["PER_b_growth"] = (final_PER_b - PER_b) / PER_b_steps
        PER_b_growth = Param["PER_b_growth"]
    else:
        Param["PER_b_growth"] = 0.0
        PER_b_growth = Param["PER_b_growth"]

    if PER_a_anneal:
        Param["PER_a_growth"] = (final_PER_a - PER_a) / PER_a_steps
        PER_a_growth = Param["PER_a_growth"]
    else:
        Param["PER_a_growth"] = 0.0
        PER_a_growth = Param["PER_a_growth"]

    if save_ckpt_model:
        save_ckpt_steps = N_train / save_ckpt_model
        Param["save_ckpt_steps"] = save_ckpt_steps

    # 1. PATH FOR MODEL (CKPT) AND TENSORBOARD OUTPUT, STORE CONFIG FILE ---------------
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

    # 2. SIMULATE SYNTHETIC DATA --------------------------------------------------------------
    returns, factors, f_speed = ReturnSampler(
        N_train,
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

    # 3. INSTANTIATE MARKET ENVIRONMENTS --------------------------------------------------------------
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
    )

    action_limit = action_quantiles[0]
    Param["action_limit"] = action_quantiles[0]

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
            action_limit=action_quantiles[0],
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
            action_limit=action_quantiles[0],
        )

    # 4. CREATE INITIAL STATE AND NETWORKS ----------------------------------------------------------
    # instantiate the initial state (return, holding) for DDPG

    CurrState, CurrFactor = env.reset()
    # instantiate the initial state for the benchmark
    if executeGP:
        CurrOptState = env.opt_reset()
        OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
    # instantiate the initial state for the markovitz solution
    if executeMV:
        CurrMVState = env.opt_reset()

    # iteration count to decide when copying weights for the Target Network
    iters = 0
    if recurrent_env:
        num_states = CurrState.shape[-1]
        num_actions = int(num_states / 2)
    else:
        num_states = len(CurrState)
        num_actions = int(num_states / 2)

    # create train and target network
    TrainNet = DDPG(
        seed_init,
        recurrent_env,
        gamma,
        max_experiences,
        update_target,
        tau_Q,
        tau_p,
        num_states,
        num_actions,
        hidden_units_Q,
        hidden_units_p,
        hidden_memory_units,
        batch_size,
        selected_loss,
        learning_rate_Q,
        learning_rate_p,
        start_train,
        optimizer_name,
        batch_norm_input,
        batch_norm_hidden,
        activation_Q,
        activation_p,
        kernel_initializer,
        plot_hist,
        plot_steps_hist,
        plot_steps,
        summary_writer,
        stddev_noise,
        theta,
        mu,
        action_limit,
        output_init,
        weight_decay_Q,
        weight_decay_p,
        delayed_actions,
        noise,
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
        clipgrad,
        clipnorm,
        clipvalue,
        clipglob_steps,
        beta_1,
        beta_2,
        eps_opt,
        lr_schedule,
        exp_decay_steps,
        exp_decay_rate_Q,
        exp_decay_rate_p,
        DDPG_type,
        noise_clip,
        stddev_pol_noise,
        rng,
        modelname="Train",
    )

    TargetNet = DDPG(
        seed_init,
        recurrent_env,
        gamma,
        max_experiences,
        update_target,
        tau_Q,
        tau_p,
        num_states,
        num_actions,
        hidden_units_Q,
        hidden_units_p,
        hidden_memory_units,
        batch_size,
        selected_loss,
        learning_rate_Q,
        learning_rate_p,
        start_train,
        optimizer_name,
        batch_norm_input,
        batch_norm_hidden,
        activation_Q,
        activation_p,
        kernel_initializer,
        plot_hist,
        plot_steps_hist,
        plot_steps,
        summary_writer,
        stddev_noise,
        theta,
        mu,
        action_limit,
        output_init,
        weight_decay_Q,
        weight_decay_p,
        delayed_actions,
        noise,
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
        clipgrad,
        clipnorm,
        clipvalue,
        clipglob_steps,
        beta_1,
        beta_2,
        eps_opt,
        lr_schedule,
        exp_decay_steps,
        exp_decay_rate_Q,
        exp_decay_rate_p,
        DDPG_type,
        noise_clip,
        stddev_pol_noise,
        rng,
        modelname="Target",
    )

    logging.info(
        "Successfully initialized Networks...YOU ARE CURRENTLY USING A SEED TO INITIALIZE WEIGHTS. LEAVE IT IF YOU HAVE FOUND A PROPER NN SETTING"
    )

    # 5. TRAIN ALGORITHM ----------------------------------------------------------
    for i in tqdm(iterable=range(N_train + 1), desc="Training DQNetwork"):

        if executeDRL:
            if i <= start_train:
                shares_traded = TrainNet.uniform_action()
            else:
                stddev_noise = max(0.0, stddev_noise - stddev_noise_decay)
                TrainNet.action_noise.sigma = stddev_noise
                shares_traded = TrainNet.noisy_action(CurrState)

            NextState, Result, NextFactors = env.step(
                CurrState, shares_traded, i, tag="DDPG"
            )
            env.store_results(Result, i)
            exp = {
                "s": CurrState,
                "a": shares_traded,
                "r": Result["Reward_DDPG"],
                "s2": NextState,
                "f": NextFactors,
            }

            TrainNet.add_experience(exp)
            TrainNet.train(TargetNet, i)
            CurrState = NextState
            iters += 1
            if (iters % copy_step == 0) and (i > TrainNet.start_train):
                TargetNet.copy_weights_Q(TrainNet)
                TargetNet.copy_weights_p(TrainNet)

            if (
                save_ckpt_model
                and (i % save_ckpt_steps == 0)
                and (i > TrainNet.start_train)
            ):
                if DDPG_type == "TD3":
                    TrainNet.Q1_model.save_weights(
                        os.path.join(
                            savedpath, "ckpt", "Q1_model_{}_it_weights".format(i)
                        ),
                        save_format="tf",
                    )
                    TrainNet.Q2_model.save_weights(
                        os.path.join(
                            savedpath, "ckpt", "Q2_model_{}_it_weights".format(i)
                        ),
                        save_format="tf",
                    )
                else:
                    TrainNet.Q_model.save_weights(
                        os.path.join(
                            savedpath, "ckpt", "Q_model_{}_it_weights".format(i)
                        ),
                        save_format="tf",
                    )

                TrainNet.p_model.save_weights(
                    os.path.join(savedpath, "ckpt", "p_model_{}_it_weights".format(i)),
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
            if (i % save_ckpt_steps == 0) and (i != 0) and (i > TrainNet.start_train):

                Out_sample_test(
                    N_test,
                    sigmaf,
                    f0,
                    f_param,
                    sigma,
                    plot_inputs,
                    HalfLife,
                    Startholding,
                    CostMultiplier,
                    kappa,
                    discount_rate,
                    executeDRL,
                    None,
                    executeMV,
                    None,
                    KLM,
                    executeGP,
                    TrainNet,
                    savedpath,
                    i,
                    recurrent_env=recurrent_env,
                    unfolding=unfolding,
                    QTable=None,
                    rng=rng,
                    seed_test=seed_ret,
                    action_limit=action_limit,
                    uncorrelated=uncorrelated,
                    t_stud=t_stud,
                    tag="DDPG",
                )

    logging.info("Successfully trained the DDPG...")
    # 6. STORE RESULTS ----------------------------------------------------------
    if not out_of_sample_test:
        Param["iterations"] = [
            str(int(i)) for i in np.arange(0, N_train + 1, save_ckpt_steps)
        ][1:]
    saveConfigYaml(Param, savedpath)
    if save_results:
        env.save_outputs(savedpath)

    if save_model:
        if DDPG_type == "TD3":
            TrainNet.Q1_model.save_weights(
                os.path.join(savedpath, "Q1_model_final_weights"), save_format="tf"
            )
            TrainNet.Q2_model.save_weights(
                os.path.join(savedpath, "Q2_model_final_weights"), save_format="tf"
            )
        else:
            TrainNet.Q_model.save_weights(
                os.path.join(savedpath, "Q_model_final_weights"), save_format="tf"
            )
        TrainNet.p_model.save_weights(
            os.path.join(savedpath, "p_model_final_weights"), save_format="tf"
        )
        logging.info("Successfully saved DDPG weights...")


if __name__ == "__main__":
    RunDDPGTraders(Param)
