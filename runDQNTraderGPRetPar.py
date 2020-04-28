# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 00:56:05 2020

@author: aless
"""

import os


if any('SPYDER' in name for name in os.environ):
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    

# 0. importing section initialize logger.--------------------------------------
import logging, os, itertools
from utils.readYaml import readConfigYaml 
from utils.generateLogger import generate_logger
from runDQNTraderGPRet import RunDQN
import pdb
# import multiprocessing
from joblib import Parallel, delayed

#0. Generate Logger-------------------------------------------------------------
logger = generate_logger()

# 1. Read config ---------------------------------------------------------------- 
# maybe substitute with argparse
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramDQNGPRet.yaml'))
assert Param['runtype'] == 'multi'
logging.info('Successfully read config file with parameters...')



variables = []
for xs in itertools.product(*[Param[v] for v in Param['varying_pars']]):
    variables.append(xs)

num_cores = len(variables)


def RunMultiParallelExp(var_par,Param):
    
    for i in range(len(var_par)):
        Param[Param['varying_pars'][i]] = var_par[i]
    
    RunDQN(Param)


if __name__ == "__main__":
    Parallel(n_jobs=num_cores)(delayed(RunMultiParallelExp)(var_par,Param) for var_par in variables)