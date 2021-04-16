# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:34:30 2020

@author: alessiobrini
"""

import os

if any("SPYDER" in name for name in os.environ):
    from IPython import get_ipython

    get_ipython().magic("reset -sf")

import os, re, logging, sys, pdb, random
from utils.common import readConfigYaml, saveConfigYaml, generate_logger, format_tousands, set_size
import numpy as np
import pandas as pd
from utils. test import Out_sample_test, Out_sample_Misspec_test
from utils.plot import (
    load_DQNmodel,
    plot_multitest_overlap_OOS,
    plot_abs_pnl_OOS,
)
 
import matplotlib.pyplot as plt
import tensorflow as tf

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


def runMultiTestPlots(p):

    N_test = p["N_test"]
    outputClass = p["outputClass"]
    outputModel = p["outputModel"]
    tag = p["algo"]


    colors = ['blue','red','green','black']
    # random.seed(2212) #7156
    # for _ in range(len(outputModel)):
    #     r = random.random()
    #     b = random.random()
    #     g = random.random()
    #     color = (r, g, b)
    #     colors.append(color)

    for t in tag:

        var_plot = [
            "NetPnl_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "Reward_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "SR_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "PnLstd_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            # "Pdist_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
            'AbsNetPnl_OOS_{}_{}.parquet.gzip'.format(format_tousands(N_test), t),
            # 'AbsRew_OOS_{}_{}.parquet.gzip'.format(format_tousands(N_test), t)
            "AbsHold_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            # 'AbsSR_OOS_{}_{}.parquet.gzip'.format(format_tousands(N_test), t),
        ]

            
        for it,v in enumerate(var_plot):

            # read main folder
            fig = plt.figure(figsize=set_size(width=1000.0))
            # fig.subplots_adjust(wspace=0.2, hspace=0.6)
            ax = fig.add_subplot()
            for k, out_mode in enumerate(outputModel):
                
                modelpath = "outputs/{}/{}".format(outputClass, out_mode)
                
                # get the latest created folder "length"
                all_subdirs = [os.path.join(modelpath,d) for d in os.listdir(modelpath) if os.path.isdir(os.path.join(modelpath,d))]
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                length = os.path.split(latest_subdir)[-1]
                
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
                # pdb.set_trace()
                
                
                if "Abs" in v:
                    
                    dfs_opt = []
                    for exp in filtered_dir:
                        exp_path = os.path.join(data_dir, exp)
                        df_opt = pd.read_parquet(os.path.join(exp_path, v.replace(t,'GP')))
                        dfs_opt.append(df_opt)
                    dataframe_opt = pd.concat(dfs_opt)
                    dataframe_opt.index = range(len(dfs_opt))
                    
                    plot_abs_pnl_OOS(
                        ax,
                        dataframe,
                        dataframe_opt,
                        data_dir,
                        N_test,
                        v,
                        colors=colors[k],
                        params=p_mod,
                        i=it
                    )

                    value = dataframe_opt.iloc[1,4]
                    std =250000
                    ax.set_ylim(value-std,value+std)
                    # pdb.set_trace()
                    
                else:
                    plot_multitest_overlap_OOS(
                        ax,
                        dataframe,
                        data_dir,
                        N_test,
                        v,
                        colors=colors[k],
                        params=p_mod,
                    )
                logging.info("Plot saved successfully...")


if __name__ == "__main__":
    runMultiTestPlots(p)
