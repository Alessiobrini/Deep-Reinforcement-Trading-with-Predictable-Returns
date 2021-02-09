# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:04:36 2020

@author: aless
"""

import os

if any("SPYDER" in name for name in os.environ):
    from IPython import get_ipython

    get_ipython().magic("reset -sf")

import os, re, logging
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from natsort import natsorted
from utils.common import (
    readConfigYaml,
    saveConfigYaml,
    generate_logger,
    format_tousands,
)
from utils.plot import (
    load_DQNmodel,
    load_Actor_Critic,
    Out_sample_test,
    Out_sample_Misspec_test,
    TrainedQTable,
)


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
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
    length = p["length"]


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

    # flag to extract checkpoints
    extract_iterations = True
    # iterate over experiments
    for exp in filtered_dir:
        exp_path = os.path.join(data_dir, exp)
        logging.info("Doing Multi Test OOS for {}...".format(exp))
        # path for the config file of the single experiment
        filenamep = os.path.join(data_dir, exp, "config_{}.yaml".format(length))
        p_mod = readConfigYaml(filenamep)

        # temporary correction to the config
        if "seed_init" not in p_mod:
            p_mod["seed_init"] = p_mod["seed"]

        # get iterations for loading checkpoints
        if extract_iterations:
            if p_mod["out_of_sample_test"]:
                ckpts = [
                    each
                    for each in os.listdir(
                        os.path.join(os.getcwd(), data_dir, filtered_dir[0])
                    )
                    if each.startswith("Test")
                ]
                ckpts = natsorted(ckpts)
                # get different point in time where we stored weights
                iterations = [re.findall(r"\d+", ckpt)[-1] for ckpt in ckpts]
                iterations_num = [int(float(i)) for i in iterations]
            else:
                iterations = p_mod["iterations"]
                # iterations = [str(int(i)) for i in np.arange(0,p_mod['N_train']+1,p_mod['save_ckpt_steps'])][1:]
                iterations_num = [int(float(i)) for i in iterations]
                iterations = [str(int(float(i))) for i in iterations]

            extract_iterations = False

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
        if p_mod["executeRL"] and "Q" in tag:
            mean_series_pnl_q = pd.DataFrame(index=range(1), columns=iterations)
            mean_series_rew_q = pd.DataFrame(index=range(1), columns=iterations)
            mean_series_sr_q = pd.DataFrame(index=range(1), columns=iterations)
            abs_series_pnl_q = pd.DataFrame(index=range(1), columns=iterations)
            abs_series_rew_q = pd.DataFrame(index=range(1), columns=iterations)
            abs_series_sr_q = pd.DataFrame(index=range(1), columns=iterations)
            abs_series_hold_q = pd.DataFrame(index=range(1), columns=iterations)

        # do tests for saved weights at intermediate time
        for ckpt_it in iterations_num:
            # import model and load trained weights
            if "DQN" in tag:
                model, actions = load_DQNmodel(
                    p_mod, exp_path, ckpt=True, ckpt_it=ckpt_it
                )
                p_mod["action_limit"] = None
            elif "DDPG" in tag:
                if p_mod["DDPG_type"] == "DDPG":
                    model = load_Actor_Critic(
                        p_mod, exp_path, ckpt=True, ckpt_it=ckpt_it
                    )
                elif p_mod["DDPG_type"] == "TD3":
                    model = load_Actor_Critic(
                        p_mod, exp_path, ckpt=True, ckpt_it=ckpt_it, DDPG_type="TD3"
                    )

            # import Qtable
            if p_mod["executeRL"] and "Q" in tag:
                tablefilename = "QTable{}.parquet.gzip".format(
                    format_tousands(ckpt_it)
                )
                QTable = pd.read_parquet(
                    os.path.join(os.getcwd(), data_dir, exp, "ckpt", tablefilename)
                )
                TrainedQ = TrainedQTable(QTable)

            else:
                TrainedQ = None
                p_mod["executeRL"] = False

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
                # opnl = []
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
                    ) = Out_sample_Misspec_test(
                        N_test=N_test,
                        df=None,  # adapt if you have real data
                        factor_lb=p_mod["factor_lb"],
                        Startholding=p_mod["Startholding"],
                        CostMultiplier=p_mod["CostMultiplier"],
                        kappa=p_mod["kappa"],
                        discount_rate=p_mod["discount_rate"],
                        executeDRL=p_mod["executeDRL"],
                        executeRL=p_mod[
                            "executeRL"
                        ],  # Put true if you want to OOS test also tab RL
                        executeMV=p_mod["executeMV"],
                        RT=p_mod["RT"],
                        KLM=p_mod["KLM"],
                        executeGP=p_mod["executeGP"],
                        TrainNet=model,  # loaded model
                        iteration=0,  # not useful, put 0
                        recurrent_env=p_mod["recurrent_env"],
                        unfolding=p_mod["unfolding"],
                        QTable=TrainedQ,  # Put the loaded table if you want to OOS test also tab RL
                        action_limit=p_mod["action_limit"],  # only useful for DDPG
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

                if p_mod["executeRL"] and "Q" in tag:
                    mean_series_pnl_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(avg_pnls)[:, 1]
                    )
                    mean_series_rew_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(avg_rews)[:, 1]
                    )
                    mean_series_sr_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(avg_srs)[:, 1]
                    )
                    abs_series_pnl_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_pnl_rl)[:, 1]
                    )
                    abs_series_rew_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_rew_rl)[:, 1]
                    )
                    abs_series_sr_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_sr_rl)[:, 1]
                    )
                    abs_series_hold_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_hold_rl)[:, 1]
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
                    ) = Out_sample_test(
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
                        executeDRL=p_mod["executeDRL"],
                        executeRL=p_mod[
                            "executeRL"
                        ],  # Put true if you want to OOS test also tab RL
                        executeMV=p_mod["executeMV"],
                        RT=p_mod["RT"],
                        KLM=p_mod["KLM"],
                        executeGP=p_mod["executeGP"],
                        TrainNet=model,  # loaded model
                        iteration=0,  # not useful, put 0
                        recurrent_env=p_mod["recurrent_env"],
                        unfolding=p_mod["unfolding"],
                        QTable=TrainedQ,  # Put the loaded table if you want to OOS test also tab RL
                        rng=rng,
                        seed_test=s,
                        action_limit=p_mod["action_limit"],  # only useful for DDPG
                        uncorrelated=p_mod["uncorrelated"],
                        t_stud=p_mod["t_stud"],
                        variables=variables,
                        side_only = p_mod['side_only'],
                        discretization = p_mod['discretization'],
                        temp = p_mod['temp'],
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
                if p_mod["executeRL"] and "Q" in tag:
                    mean_series_pnl_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(avg_pnls)[:, 1]
                    )
                    mean_series_rew_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(avg_rews)[:, 1]
                    )
                    mean_series_sr_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(avg_srs)[:, 1]
                    )
                    abs_series_pnl_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_pnl_rl)[:, 1]
                    )
                    abs_series_rew_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_rew_rl)[:, 1]
                    )
                    abs_series_sr_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_sr_rl)[:, 1]
                    )
                    abs_series_hold_q.loc[0, str(ckpt_it)] = np.mean(
                        np.array(abs_hold_rl)[:, 1]
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

        if p_mod["executeRL"] and "Q" in tag:
            mean_series_pnl_q.to_parquet(
                os.path.join(
                    exp_path,
                    "NetPnl_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
                ),
                compression="gzip",
            )
            mean_series_rew_q.to_parquet(
                os.path.join(
                    exp_path,
                    "Reward_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
                ),
                compression="gzip",
            )
            mean_series_sr_q.to_parquet(
                os.path.join(
                    exp_path,
                    "SR_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
                ),
                compression="gzip",
            )
            abs_series_pnl_q.to_parquet(
                os.path.join(
                    exp_path,
                    "AbsNetPnl_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
                ),
                compression="gzip",
            )
            abs_series_rew_q.to_parquet(
                os.path.join(
                    exp_path,
                    "AbsReward_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
                ),
                compression="gzip",
            )
            abs_series_sr_q.to_parquet(
                os.path.join(
                    exp_path,
                    "AbsSR_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
                ),
                compression="gzip",
            )
            abs_series_hold_q.to_parquet(
                os.path.join(
                    exp_path,
                    "AbsHold_OOS_{}_{}.parquet.gzip".format(
                        format_tousands(N_test), tag[1]
                    ),
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

    end = time.time()
    logging.info("Script has finished in {} minutes".format((end - start) / 60))

if __name__ == "__main__":
    runMultiTestOOS(p)
