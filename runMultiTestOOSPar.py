# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:25:28 2020

@author: aless
"""
import logging, os, itertools, sys
from utils.common import readConfigYaml, generate_logger, chunks
from runMultiTestOOS import runMultiTestOOS
from runMultiTestOOSParbySeed import runMultiTestOOSbySeed
from itertools import combinations
import numpy as np
import time
from joblib import Parallel, delayed


# 0. Generate Logger-------------------------------------------------------------
logger = generate_logger()

# 1. Read config ----------------------------------------------------------------
Param = readConfigYaml(os.path.join(os.getcwd(), "config", "paramMultiTestOOS.yaml"))
logging.info("Successfully read config file with parameters...")


variables = []

if Param["varying_type"] == "combination":
    for xs in itertools.product(*[Param[v] for v in Param["varying_pars"]]):
        variables.append(xs)
elif Param["varying_type"] == "ordered_combination":
    for xs in zip(*[Param[v] for v in Param["varying_pars"]]):
        variables.append(xs)
elif Param["varying_type"] == "subset_combination":
    maxsubset = len(Param[Param["varying_pars"][0]])
    for i in range(maxsubset):
        iterables = [
            combinations(Param[param_name], i + 1)
            for param_name in Param["varying_pars"]
        ]
        for tup in zip(*iterables):
            variables.append([list(tup_el) for tup_el in tup])

elif Param["varying_type"] == "random_search":
    for xs in itertools.product(*[Param[v] for v in Param["varying_pars"]]):
        variables.append(xs)
    variables = [
        variables[i]
        for i in np.random.randint(0, len(variables) - 1, Param["num_rnd_search"])
    ]
elif Param["varying_type"] == "chunk":
    for xs in itertools.product(*[Param[v] for v in Param["varying_pars"]]):
        variables.append(xs)

else:
    print("Choose proper way to combine varying parameters")
    sys.exit()

num_cores = len(variables)


def RunMultiParallelExp(var_par, Param):

    for i in range(len(var_par)):
        Param[Param["varying_pars"][i]] = var_par[i]

    # decide wether to parallelize by seeds or not
    if Param["parbyseed"]:
        runMultiTestOOSbySeed(Param)
    else:
        runMultiTestOOS(Param)


if __name__ == "__main__":
    if Param["varying_type"] == "random_search":
        Parallel(n_jobs=num_cores)(
            delayed(RunMultiParallelExp)(var_par, Param) for var_par in variables
        )
        time.sleep(10)
        os.execv(sys.executable, ["python"] + sys.argv)
        # os.execv(__file__, sys.argv)
    if Param["varying_type"] == "chunk":
        num_cores = Param["num_rnd_search"]
        for chunk_var in chunks(variables, num_cores):
            Parallel(n_jobs=num_cores)(
                delayed(RunMultiParallelExp)(var_par, Param) for var_par in chunk_var
            )
            # time.sleep(10)
    else:
        Parallel(n_jobs=num_cores)(
            delayed(RunMultiParallelExp)(var_par, Param) for var_par in variables
        )
