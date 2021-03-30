# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:22:38 2021

@author: alessiobrini
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
import torch
import torch.nn as nn
import numpy as np
from utils.common import (
    generate_logger,
    readConfigYaml,
    saveConfigYaml,
    GeneratePathFolder,
)
import pandas as pd
from utils.simulation import ReturnSampler, create_lstm_tensor, GARCHSampler
from utils.env import (
    MarketEnv,
    RecurrentMarketEnv,
    ActionSpace,
)
from utils.PPO import PPO
from utils.tools import get_action_boundaries
from utils.math_tools import unscale_action
from utils.tools import CalculateLaggedSharpeRatio, RunModels


# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMisspecPPO.yaml"))
logging.info("Successfully read config file with parameters...")


def RunMisspecPPOTraders(p):
    """Main function which loads the parameters for the synthetic experiments
    and run both training and testing routines

    parameters
    ----------
    p: dict
        The dictionary containing the parameters
    """
    # 0. EXTRACT parameters ----------------------------------------------------------
    # RL related
    universal = p['universal']
    policy_type = p['policy_type']
    pol_std = p['pol_std']
    recurrent_env = p["recurrent_env"]
    gamma = p["gamma"]
    kappa = p["kappa"]
    tau = p['tau']
    clip_param = p['clip_param']
    vf_c = p['vf_c']
    ent_c = p['ent_c']
    qts = p["qts"]
    KLM = p["KLM"]
    zero_action = p["zero_action"]
    min_n_actions = p["min_n_actions"]
    side_only = p['side_only']
    # DL related
    activation = p["activation"]
    optimizer_name = p["optimizer_name"]
    beta_1 = p["beta_1"]
    beta_2 = p["beta_2"]
    eps_opt = p["eps_opt"]
    hidden_units_value = p['hidden_units_value']
    hidden_units_actor = p['hidden_units_actor']
    batch_size = p["batch_size"]
    learning_rate = p["learning_rate"]
    lr_schedule = p["lr_schedule"]
    exp_decay_pct = p["exp_decay_pct"]
    exp_decay_rate = p["exp_decay_rate"]
    # hidden_memory_units = p["hidden_memory_units"]
    unfolding = p["unfolding"]
    # Regularization
    batch_norm_input = p["batch_norm_input"]
    batch_norm_value_out = p['batch_norm_value_out']
    # Data Simulation
    datatype = p["datatype"]
    factor_lb = p["factor_lb"]
    HalfLife = p["HalfLife"]
    f0 = p["f0"]
    f_param = p["f_param"]
    sigma = p["sigma"]
    sigmaf = p["sigmaf"]
    uncorrelated = p["uncorrelated"]
    CostMultiplier = p["CostMultiplier"]
    discount_rate = p["discount_rate"]
    Startholding = p["Startholding"]
    # Experiment and storage
    ppo_epochs = p['ppo_epochs']
    episodes = p["episodes"]
    len_series = p['len_series']
    seed = p["seed"]
    seed_param = p["seed_param"]
    plot_inputs = p["plot_inputs"]
    save_results = p["save_results"]
    save_ckpt_model = p["save_ckpt_model"]
    outputDir = p["outputDir"]
    outputClass = p["outputClass"]
    outputModel = p["outputModel"]
    varying_pars = p["varying_pars"]
    

    len_series = p['len_series']
    mean_process = p["mean_process"]
    lags_mean_process = p["lags_mean_process"]
    vol_process = p["vol_process"]
    distr_noise = p["distr_noise"]
    HalfLife = p["HalfLife"]
    f0 = p["f0"]
    f_param = p["f_param"]
    sigma = p["sigma"]
    sigmaf = p["sigmaf"]
    degrees = p["degrees"]

    # set random number generator
    rng = np.random.RandomState(seed)

    if not recurrent_env:
        p["unfolding"] = unfolding = 1
        
    N_train = len_series * episodes
    p['N_train'] = N_train
    
    # 1. SIMULATE SYNTHETIC DATA --------------------------------------------------------------
    
    if datatype == "garch":
        return_series, params = GARCHSampler(
            len_series + factor_lb[-1] + unfolding + 1,
            mean_process=mean_process,
            lags_mean_process=lags_mean_process,
            vol_process=vol_process,
            distr_noise=distr_noise,
            seed=seed,
            seed_param=seed_param,
        )
        params = {"_".join([datatype, k]): v for k, v in params.items()}
        p = {**p, **params}

        df = CalculateLaggedSharpeRatio(
            return_series, factor_lb, nameTag=datatype, seriestype="return"
        )

        y, X = df[df.columns[0]], df[df.columns[1:]]
    elif datatype == "t_stud":
        plot_inputs = False
        # df freedom for t stud distribution are hard coded inside the function
        returns, factors, f_speed = ReturnSampler(
            len_series + factor_lb[-1],
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
                len_series + factor_lb[-1],
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
                len_series,
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
            len_series + factor_lb[-1],
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
    else:
        params_retmodel, params_meanrev, fitted_retmodel, fitted_ous = RunModels(y, X)
        
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
        p["HalfLife"] = HalfLife_fit
        p["f_speed"] = f_speed_fit
        p["f_param"] = f_param_fit
        p["sigma"] = sigma_fit
        p["sigmaf"] = sigmaf_fit
    elif datatype == "t_stud" or datatype == 'garch_mr':
        p["HalfLife_fit"] = HalfLife_fit
        p["f_speed_fit"] = f_speed_fit
        p["f_param_fit"] = f_param_fit
        p["sigma_fit"] = sigma_fit
        p["sigmaf_fit"] = sigmaf_fit
    elif datatype == "t_stud_mfit":
        p["HalfLife_fit"] = HalfLife_fit
        p["f_speed_fit"] = f_speed_fit
        p["f_param_fit"] = f_param
        p["sigma_fit"] = sigma
        p["sigmaf_fit"] = sigmaf

    # print(sigma,sigma_fit)
    # print(HalfLife,HalfLife_fit)
    # print(f_speed,f_speed_fit)
    # print(f_param,f_param_fit)
    # pdb.set_trace()
    
    # 2. SET UP SOME HYPERPARAMETERS --------------------------------------------------------------

    exp_decay_steps = int(N_train * exp_decay_pct)
    p["exp_decay_steps"] = exp_decay_steps

    if save_ckpt_model:
        save_ckpt_steps = int(episodes / save_ckpt_model)
        p["save_ckpt_steps"] = save_ckpt_steps

    if recurrent_env:
        returns_tens = create_lstm_tensor(returns.reshape(-1, 1), unfolding)
        factors_tens = create_lstm_tensor(factors, unfolding)
    logging.info("Successfully loaded data...")

    # 3. PATH FOR MODEL (CKPT) AND TENSORBOARD OUTPUT, STORE CONFIG FILE ---------------
    savedpath = GeneratePathFolder(
        outputDir, outputClass, outputModel, varying_pars, N_train, p
    )
    saveConfigYaml(p, savedpath)
    if save_ckpt_model and not os.path.exists(os.path.join(savedpath, "ckpt")):
        os.makedirs(os.path.join(savedpath, "ckpt"))
    elif save_ckpt_model and os.path.exists(os.path.join(savedpath, "ckpt")):
        pass
    logging.info("Successfully generated path and stored config...")



    # 4. INSTANTIATE MARKET ENVIRONMENT --------------------------------------------------------------
    
    action_quantiles, ret_quantile, holding_quantile = get_action_boundaries(
        HalfLife,
        Startholding,
        sigma,
        CostMultiplier,
        kappa,
        len_series,
        discount_rate,
        f_param,
        f_speed_fit,
        returns,
        factors,
        qts=qts,
        min_n_actions=min_n_actions,
    )
    
    KLM[:2] = action_quantiles
    KLM[2] = holding_quantile
    action_limit = KLM[0]
    p["KLM"] = KLM
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
            f_speed,
            returns,
            factors,
            returns_tens,
            factors_tens,
            action_limit=KLM[0]
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
            action_limit=KLM[0]
        )

    logging.info("Successfully initialized the market environment...")

    # 4. CREATE INITIAL STATE AND NETWORKS ----------------------------------------------------------
    input_shape = env.get_state_dim()
    p['n_inputs'] = input_shape[0]
    # create train and target network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    PPO_ = PPO(
        seed,
        gamma,
        tau,
        clip_param,
        vf_c,
        ent_c,
        input_shape,
        hidden_units_value,
        hidden_units_actor,
        batch_size,
        learning_rate,
        activation,
        optimizer_name,
        batch_norm_input,
        batch_norm_value_out,
        action_space,
        policy_type,
        pol_std,
        beta_1,
        beta_2,
        eps_opt,
        lr_schedule,
        exp_decay_rate,
        rng,
    )

    PPO_.model.to(device)

    logging.info(
        "Successfully initialized the PPO model..."
    )


    # 5. TRAIN ALGORITHM ----------------------------------------------------------
    iterations = []
    iters = 0
    for e in tqdm(iterable=range(episodes), desc='Running episodes...'):
        
        if universal:
            # TODO fix universal
            returns, factors, f_speed = ReturnSampler( len_series,
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
                                                        disable_tqdm=True,
                                                    )
            env.returns = returns
            env.factors = factors
        
        state, _ = env.reset()
        PPO_.reset_experience()
        # if executeGP:
        #     CurrOptState = env.opt_reset()
        #     OptRate, DiscFactorLoads = env.opt_trading_rate_disc_loads()
        
        for i in range(len_series - 1):
            
            # TODO implement case in which you output also qvalues and can calculate betsize
            dist, value = PPO_.act(state)

            if policy_type == 'continuous':
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                clipped_action = nn.Tanh()(action).cpu().numpy().ravel()
                action = action.cpu().numpy().ravel()

                unscaled_action = unscale_action(action_limit,clipped_action)

            elif policy_type == 'discrete':
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                clipped_action = np.array([action_space.values[action]], dtype=np.float32)
                unscaled_action = clipped_action
                action = np.array([action], dtype=np.float32)
                
            else:
                print('Select a policy as continuous or discrete')
                sys.exit()
            
            # TODO eventually introduce bet size and bcm
            # if bcm and side_only:
                
            #     _, OptResult = env.opt_step(
            #         CurrOptState, OptRate, DiscFactorLoads, i
            #         )
                
            #     exp_bcm = {"unsc_a": unscaled_shares_traded,
            #                "opt_a": OptResult['OptNextAction']}
            #     exp = {**exp, **exp_bcm}
                
            # elif bcm and not side_only:
                
            #     _, OptResult = env.opt_step(
            #         CurrOptState, OptRate, DiscFactorLoads, i
            #         )
                
            #     exp_bcm = {"opt_a": OptResult['OptNextAction']}   
            #     exp = {**exp, **exp_bcm}

 
            next_state, Result, _ = env.step(state, unscaled_action[0], i, tag='PPO')

            exp = {"state": state, 
                   "action": action, 
                   "reward": Result['Reward_PPO'], 
                   "log_prob": log_prob.detach().cpu().numpy().ravel(), # avoid require_grad and go back to numpy array
                   "value": value.detach().cpu().numpy().ravel(),
                   }
            
            PPO_.add_experience(exp)
            
            state = next_state
            iters += 1
        
        # next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = PPO_.act(next_state)
        # compute the advantage estimate from the given rollout
        PPO_.compute_gae(next_value.detach().cpu().numpy().ravel())
        
        for _ in range(ppo_epochs): # run for more than one epochs
            for state, action, old_log_probs, return_, advantage in PPO_.ppo_iter():

                PPO_.train(state, action, old_log_probs, return_, advantage)
                
            # recompute gae to avoid stale advantages
            if _ == len(range(ppo_epochs)) -1 :
                pass
            else:
                PPO_.compute_gae(next_value.detach().cpu().numpy().ravel(),
                                 recompute_value=True)
                
        # store weights every tot episodes
        if (
            save_ckpt_model
            and (e % save_ckpt_steps == 0)
        ):
            torch.save(
                PPO_.model.state_dict(),
                os.path.join(savedpath, "ckpt", "PPO_{}_ep_weights.pth".format(e+1)),
            )
            
            iterations.append(str(e+1))
    
        elif(
            save_ckpt_model
            and (e == range(episodes)[-1])
        ):
            
            torch.save(
                PPO_.model.state_dict(),
                os.path.join(savedpath, "ckpt", "PPO_{}_ep_weights.pth".format(e+1)),
            )
            
            iterations.append(str(e+1))

    logging.info("Successfully trained the PPO policy...") 
                
    # 6. STORE RESULTS ----------------------------------------------------------
    p["iterations"] = iterations
    saveConfigYaml(p, savedpath)
    if save_results:
        env.save_outputs(savedpath)

if __name__ == "__main__":
    RunMisspecPPOTraders(p)
