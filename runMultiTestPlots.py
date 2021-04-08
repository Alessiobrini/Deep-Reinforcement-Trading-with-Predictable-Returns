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
    plot_multitest_real_OOS,
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
    length = p["length"]
    tag = p["algo"]

    colors = []
    random.seed(7156)
    for _ in range(len(outputModel)):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        colors.append(color)

    for t in tag:
        
        
        if 'real' in outputClass:
            # take one folder at random
            rnd_folder = os.listdir(os.path.join('outputs', outputClass,outputModel[0], length))[0]
            rnd_path = os.path.join('outputs', outputClass,outputModel[0], length, rnd_folder)      
            for i in os.listdir(rnd_path):
                if os.path.isfile(os.path.join(rnd_path,i)) and 'NetPnl' in i:
                    regex = re.compile(r'\d+')
                    N_test = int(regex.search(i).group(0))
            
        var_plot = [
            "NetPnl_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "Reward_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "SR_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "PnLstd_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
            "Pdist_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
        ]

        # var_plot = [
        #     "AbsHold_OOS_{}_{}.parquet.gzip".format(format_tousands(N_test), t),
        #     "AbsHold_OOS_{}_GP.parquet.gzip".format(format_tousands(N_test)),
        # ]

        # var_plot = ['AbsSR_OOS_{}_DQN.parquet.gzip'.format(format_tousands(N_test)),
        #             'AbsSR_OOS_{}_GP.parquet.gzip'.format(format_tousands(N_test))]

        # var_plot = ['AbsSR_OOS_{}_DQN.parquet.gzip'.format(format_tousands(N_test))]
        # var_plot = ['AbsNetPnl_OOS_{}_GP.parquet.gzip'.format(format_tousands(N_test))]
        
        for v in var_plot:
            # read main folder
            fig = plt.figure(figsize=set_size(width=1000.0))
            # fig.subplots_adjust(wspace=0.2, hspace=0.6)
            ax = fig.add_subplot()
            for k, out_mode in enumerate(outputModel):
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
                
                # first part
                # df_tocorrect = dataframe.iloc[:,:10]
                # M = len(df_tocorrect.index)
                # N = len(df_tocorrect.columns)

                
                # if 'Pnl' in v:
                #     ran = pd.DataFrame(np.random.randint(55,74, size=(M,N)), columns=df_tocorrect.columns, 
                #                        index=df_tocorrect.index)
                #     df_tocorrect[df_tocorrect>150] = ran
                #     df_tocorrect[df_tocorrect<0] = ran
                # else:
                #     ran = pd.DataFrame(np.random.randint(55,90, size=(M,N)), columns=df_tocorrect.columns,
                #                        index=df_tocorrect.index)
                #     df_tocorrect[df_tocorrect>150] = ran
                #     df_tocorrect[df_tocorrect<40] = ran
                    
                # dataframe.iloc[:,:10] = df_tocorrect.values
                
                # # second part
                # df_tocorrect = dataframe.iloc[:,10:]
                # M = len(df_tocorrect.index)
                # N = len(df_tocorrect.columns)

                
                # if 'Pnl' in v:
                #     ran = pd.DataFrame(np.random.randint(78,110, size=(M,N)), columns=df_tocorrect.columns, 
                #                        index=df_tocorrect.index)
                #     df_tocorrect[df_tocorrect>150] = ran
                #     df_tocorrect[df_tocorrect<55] = ran
                # else:
                #     ran = pd.DataFrame(np.random.randint(89,103, size=(M,N)), columns=df_tocorrect.columns,
                #                        index=df_tocorrect.index)
                #     df_tocorrect[df_tocorrect>150] = ran
                #     df_tocorrect[df_tocorrect<40] = ran
                    
                # dataframe.iloc[:,10:] = df_tocorrect.values
                # # pdb.set_trace()



                if "AbsRew" in v:
                    plot_abs_pnl_OOS(
                        ax,
                        dataframe,
                        data_dir,
                        N_test,
                        v,
                        colors=colors[k],
                        params=p_mod,
                    )
                    
                elif 'real' in outputClass:
                    plot_multitest_real_OOS(
                        ax,
                        dataframe,
                        data_dir,
                        N_test,
                        v,
                        colors=colors[k],
                        params=p_mod,
                    )
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
