# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:26:21 2021

@author: alessiobrini
"""
import os

if any("SPYDER" in name for name in os.environ):
    from IPython import get_ipython

    get_ipython().magic("reset -sf")

import os, re, logging
import numpy as np
import pandas as pd
import time
import torch
from natsort import natsorted
from utils.common import (
    readConfigYaml,
    saveConfigYaml,
    generate_logger,
    format_tousands,
)
from utils.test import Out_sample_test_PPO, Out_sample_Misspec_test
from utils.plot import (
    load_PPOmodel,
)

import pdb
import seaborn as sns

sns.set_style("darkgrid")


# Generate Logger-------------------------------------------------------------
logger = generate_logger()

# Read config ----------------------------------------------------------------
p = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))
logging.info("Successfully read config file for Multi Test OOS...")


def runMultiTestOOS(p):

    start = time.time()

    random_state = p["random_state"]
    N_test = p["N_test"]
    n_seeds = p["n_seeds"]
    outputClass = p["outputClass"]
    outputModel = p["outputModel"]
    algo = p["algo"]
    

    # get the latest created folder "length"
    modelpath = os.path.join(os.getcwd(),'outputs',outputClass,outputModel)
    all_subdirs = [os.path.join(modelpath,d) for d in os.listdir(modelpath) if os.path.isdir(os.path.join(modelpath,d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    length = os.path.split(latest_subdir)[-1]

    tag = algo
    variables = []
    for t in tag:
        variables.append("NetPNL_{}".format(t))
        variables.append("Reward_{}".format(t))
    variables.append("OptNetPNL")
    variables.append("OptReward")

    # read main folder
    data_dir = "outputs/{}/{}/{}".format(outputClass, outputModel, length)
    


    # Generate and store multi test OOS ----------------------------------------------------------------

    saveConfigYaml(p, data_dir, multitest=True)
    # get all the folder experiments inside the main folder
    filtered_dir = [
        dirname
        for dirname in os.listdir(data_dir)
        if os.path.isdir(os.path.join(os.getcwd(), data_dir, dirname))
    ]

    # iterate over experiments
    for exp in filtered_dir:
        exp_path = os.path.join(data_dir, exp)
        logging.info("Doing Multi Test OOS for {}...".format(exp))
        # path for the config file of the single experiment
        filenamep = os.path.join(data_dir, exp, "config_{}.yaml".format(length))
        p_mod = readConfigYaml(filenamep)
        iterations = p_mod["iterations"]
        # temporary correction to the config
        if "seed_init" not in p_mod:
            p_mod["seed_init"] = p_mod["seed"]


        # take seeds for OOS test and initialize Dataframe to store PnL averages
        rng = np.random.RandomState(random_state)
        seeds = rng.choice(1000, n_seeds, replace=False)
        rng = np.random.RandomState(random_state)
        mean_series_pnl = pd.DataFrame(index=range(1), columns=iterations)
        mean_series_rew = pd.DataFrame(index=range(1), columns=iterations)
        mean_series_sr = pd.DataFrame(index=range(1), columns=iterations)

        abs_series_pnl_rl = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_pnl_gp = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_rew_rl = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_rew_gp = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_sr_rl = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_sr_gp = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_rew_gp = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_hold_rl = pd.DataFrame(index=range(1), columns=iterations)
        abs_series_hold_gp = pd.DataFrame(index=range(1), columns=iterations)
        
        mean_series_pnl_std = pd.DataFrame(index=range(1), columns=iterations)
        
        mean_series_pdist = pd.DataFrame(index=range(1), columns=iterations)

        # do tests for saved weights at intermediate time
        for ckpt_it in iterations:
            # import model and load trained weights
            model, action = load_PPOmodel(p_mod, exp_path, int(ckpt_it))

            # check the type of experiment and do OOS tests accordingly
            if "datatype" in p_mod:
                avg_pnls = []
                avg_rews = []
                avg_srs = []
                abs_pnl_rl = []
                abs_pnl_gp = []
                abs_rew_rl = []
                abs_rew_gp = []
                abs_sr_rl = []
                abs_sr_gp = []
                abs_hold_rl = []
                abs_hold_gp = []
                avg_pnlstd = []
                avg_pdist = []
                for s in seeds:
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
                        pnl_std,
                        pdist
                    ) = Out_sample_Misspec_test(
                        N_test=N_test,
                        df=None,  # adapt if you have real data
                        factor_lb=p_mod["factor_lb"],
                        Startholding=p_mod["Startholding"],
                        CostMultiplier=p_mod["CostMultiplier"],
                        kappa=p_mod["kappa"],
                        discount_rate=p_mod["discount_rate"],
                        KLM=p_mod["KLM"],
                        executeGP=p_mod["executeGP"],
                        TrainNet=model,  # loaded model
                        policy_type=p_mod['policy_type'],
                        iteration=0,  # not useful, put 0
                        recurrent_env=p_mod["recurrent_env"],
                        unfolding=p_mod["unfolding"],
                        datatype=p_mod["datatype"],
                        mean_process=p_mod["mean_process"],
                        lags_mean_process=p_mod["lags_mean_process"],
                        vol_process=p_mod["vol_process"],
                        distr_noise=p_mod["distr_noise"],
                        seed=s,  # seed you are iterating over
                        seed_param=p_mod["seedparam"],
                        sigmaf=p_mod["sigmaf"],
                        f0=p_mod["f0"],
                        f_param=p_mod["f_param"],
                        sigma=p_mod["sigma"],
                        HalfLife=p_mod["HalfLife"],
                        uncorrelated=p_mod["uncorrelated"],
                        degrees=p_mod["degrees"],
                        rng=rng,  # not really useful if you pass a seed_test
                        variables=variables,
                        side_only = p_mod['side_only'],
                        discretization = p_mod['discretization'],
                        temp = p_mod['temp'],
                        zero_action=p_mod['zero_action'],
                        tag=tag,
                    )

                    avg_pnls.append(pnl)
                    avg_rews.append(rew)
                    avg_srs.append(sr)
                    abs_pnl_rl.append(abs_prl)
                    abs_pnl_gp.append(abs_pgp)
                    abs_rew_rl.append(abs_rewrl)
                    abs_rew_gp.append(abs_rewgp)
                    abs_sr_rl.append(abs_srrl)
                    abs_sr_gp.append(abs_srgp)
                    abs_hold_rl.append(abs_hold)
                    abs_hold_gp.append(abs_opthold)
                    avg_pnlstd.append(pnl_std)
                    avg_pdist.append(pdist)

                # append the average cumulative pnl obtained
                mean_series_pnl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(avg_pnls)[:, 0]
                )
                mean_series_rew.loc[0, str(ckpt_it)] = np.mean(
                    np.array(avg_rews)[:, 0]
                )
                mean_series_sr.loc[0, str(ckpt_it)] = np.mean(
                    np.array(avg_srs)[:, 0]
                )
                abs_series_pnl_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_pnl_rl)[:, 0]
                )
                abs_series_pnl_gp.loc[0, str(ckpt_it)] = np.mean(abs_pnl_gp)
                abs_series_rew_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_rew_rl)[:, 0]
                )
                abs_series_rew_gp.loc[0, str(ckpt_it)] = np.mean(abs_rew_gp)
                abs_series_sr_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_sr_rl)[:, 0]
                )
                abs_series_sr_gp.loc[0, str(ckpt_it)] = np.mean(abs_sr_gp)
                abs_series_hold_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_hold_rl)[:, 0]
                )
                abs_series_hold_gp.loc[0, str(ckpt_it)] = np.mean(abs_hold_gp)
                
                mean_series_pnl_std.loc[0, str(ckpt_it)] = np.mean(
                                np.array(avg_pnlstd)[:, 0]
                            )

                mean_series_pdist.loc[0, str(ckpt_it)] = np.mean(
                                np.array(avg_pdist)[:, 0]
                            )

            else:
                avg_pnls = []
                avg_rews = []
                avg_srs = []
                abs_pnl_rl = []
                abs_pnl_gp = []
                abs_rew_rl = []
                abs_rew_gp = []
                abs_sr_rl = []
                abs_sr_gp = []
                abs_hold_rl = []
                abs_hold_gp = []
                avg_pnlstd = []
                avg_pdist = []
                for s in seeds:
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
                        pnl_std,
                        pdist
                    ) = Out_sample_test_PPO(
                        N_test=N_test,
                        sigmaf=p_mod["sigmaf"],
                        f0=p_mod["f0"],
                        f_param=p_mod["f_param"],
                        sigma=p_mod["sigma"],
                        plot_inputs=False,
                        HalfLife=p_mod["HalfLife"],
                        Startholding=p_mod["Startholding"],
                        CostMultiplier=p_mod["CostMultiplier"],
                        kappa=p_mod["kappa"],
                        discount_rate=p_mod["discount_rate"],
                        KLM=p_mod["KLM"],
                        executeGP=p_mod["executeGP"],
                        TrainNet=model,  # loaded model
                        policy_type = p_mod['policy_type'],
                        recurrent_env=p_mod["recurrent_env"],
                        unfolding=p_mod["unfolding"],
                        rng=rng,
                        seed_test=s,
                        uncorrelated=p_mod["uncorrelated"],
                        t_stud=p_mod["t_stud"],
                        variables=variables,
                        side_only = p_mod['side_only'],
                        discretization = p_mod['discretization'],
                        temp = p_mod['temp'],
                        zero_action=p_mod['zero_action'],
                        tag=tag,
                    )
                        
                    avg_pnls.append(pnl)
                    avg_rews.append(rew)
                    avg_srs.append(sr)
                    abs_pnl_rl.append(abs_prl)
                    abs_pnl_gp.append(abs_pgp)
                    abs_rew_rl.append(abs_rewrl)
                    abs_rew_gp.append(abs_rewgp)
                    abs_sr_rl.append(abs_srrl)
                    abs_sr_gp.append(abs_srgp)
                    abs_hold_rl.append(abs_hold)
                    abs_hold_gp.append(abs_opthold)
                    avg_pnlstd.append(pnl_std)
                    avg_pdist.append(pdist)

                # append the average cumulative pnl obtained

                mean_series_pnl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(avg_pnls)[:, 0]
                )
                mean_series_rew.loc[0, str(ckpt_it)] = np.mean(
                    np.array(avg_rews)[:, 0]
                )
                mean_series_sr.loc[0, str(ckpt_it)] = np.mean(
                    np.array(avg_srs)[:, 0]
                )
                abs_series_pnl_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_pnl_rl)[:, 0]
                )
                abs_series_pnl_gp.loc[0, str(ckpt_it)] = np.mean(abs_pnl_gp)
                abs_series_rew_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_rew_rl)[:, 0]
                )
                abs_series_rew_gp.loc[0, str(ckpt_it)] = np.mean(abs_rew_gp)
                abs_series_sr_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_sr_rl)[:, 0]
                )
                abs_series_sr_gp.loc[0, str(ckpt_it)] = np.mean(abs_sr_gp)
                abs_series_hold_rl.loc[0, str(ckpt_it)] = np.mean(
                    np.array(abs_hold_rl)[:, 0]
                )
                abs_series_hold_gp.loc[0, str(ckpt_it)] = np.mean(abs_hold_gp)
                
                mean_series_pnl_std.loc[0, str(ckpt_it)] = np.mean(
                                    np.array(avg_pnlstd)[:, 0]
                                )
                
                mean_series_pdist.loc[0, str(ckpt_it)] = np.mean(
                                np.array(avg_pdist)[:, 0]
                            )                
                
     
        mean_series_pnl.to_parquet(
            os.path.join(
                exp_path,
                "NetPnl_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag[0]
                ),
            ),
            compression="gzip",
        )
        mean_series_rew.to_parquet(
            os.path.join(
                exp_path,
                "Reward_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag[0]
                ),
            ),
            compression="gzip",
        )
        mean_series_sr.to_parquet(
            os.path.join(
                exp_path,
                "SR_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), tag[0]),
            ),
            compression="gzip",
        )
        
        mean_series_pnl_std.to_parquet(
            os.path.join(
                exp_path,
                "PnLstd_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), tag[0]),
            ),
            compression="gzip",
        )


        abs_series_pnl_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsNetPnl_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag[0]
                ),
            ),
            compression="gzip",
        )
        abs_series_pnl_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsNetPnl_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        abs_series_rew_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsRew_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag[0]
                ),
            ),
            compression="gzip",
        )
        abs_series_rew_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsRew_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        abs_series_sr_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsSR_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag[0]
                ),
            ),
            compression="gzip",
        )
        abs_series_sr_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsSR_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        abs_series_hold_rl.to_parquet(
            os.path.join(
                exp_path,
                "AbsHold_OOS_{}_{}.parquet.gzip".format(
                    format_tousands(N_test), tag[0]
                ),
            ),
            compression="gzip",
        )
        abs_series_hold_gp.to_parquet(
            os.path.join(
                exp_path,
                "AbsHold_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )
        
        mean_series_pdist.to_parquet(
            os.path.join(
                exp_path,
                "Pdist_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            ),
            compression="gzip",
        )

    end = time.time()

    logging.info("Script has finished in {} minutes".format((end - start) / 60))

if __name__ == "__main__":
    runMultiTestOOS(p)
