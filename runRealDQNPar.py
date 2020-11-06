# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:33:38 2020

@author: aless
"""
import os


if any('SPYDER' in name for name in os.environ):
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    

# 0. importing section initialize logger.--------------------------------------
import logging, os, itertools, sys
from utils.readYaml import readConfigYaml 
from utils.generateLogger import generate_logger
from runRealDQN import RunRealDQNTraders
import pdb
from itertools import combinations
import numpy as np
import time
# import multiprocessing
from joblib import Parallel, delayed

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#0. Generate Logger-------------------------------------------------------------
logger = generate_logger()

# 1. Read config ---------------------------------------------------------------- 
# maybe substitute with argparse
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramRealDQN.yaml'))
assert Param['runtype'] == 'multi'
logging.info('Successfully read config file with parameters...')



variables = []


if Param['varying_type'] == 'combination':
    for xs in itertools.product(*[Param[v] for v in Param['varying_pars']]):
        variables.append(xs)
elif Param['varying_type'] == 'ordered_combination':
    for xs in zip(*[Param[v] for v in Param['varying_pars']]):
        variables.append(xs)   
elif Param['varying_type'] == 'subset_combination':
    maxsubset = len(Param[Param['varying_pars'][0]])
    for i in range(maxsubset):
        iterables = [combinations(Param[param_name],i+1) for param_name in Param['varying_pars']]
        for tup in zip(*iterables): 
            variables.append([list(tup_el) for tup_el in tup])
            
elif Param['varying_type'] == 'random_search':
    for xs in itertools.product(*[Param[v] for v in Param['varying_pars']]):
        variables.append(xs)
    variables = [variables[i] for i in np.random.randint(0,len(variables)-1,Param['num_rnd_search'])]
elif Param['varying_type'] == 'chunk':
    for xs in itertools.product(*[Param[v] for v in Param['varying_pars']]):
        variables.append(xs)
    
else: 
    print('Choose proper way to combine varying parameters')
    sys.exit()
    
num_cores = len(variables)

def RunMultiParallelExp(var_par,Param):
    
    for i in range(len(var_par)):
        Param[Param['varying_pars'][i]] = var_par[i]
    
    RunRealDQNTraders(Param)


if __name__ == "__main__":
    if Param['varying_type'] == 'random_search':
        Parallel(n_jobs=num_cores)(delayed(RunMultiParallelExp)(var_par,Param) for var_par in variables)
        time.sleep(300)
        os.execv(sys.executable, ['python']+ sys.argv)
        # os.execv(__file__, sys.argv)
    elif Param['varying_type'] == 'chunk':
        num_cores = Param['num_rnd_search']
        for chunk_var in chunks(variables,num_cores):
            Parallel(n_jobs=num_cores)(delayed(RunMultiParallelExp)(var_par,Param) for var_par in chunk_var)
            time.sleep(60)
    else:
        Parallel(n_jobs=num_cores)(delayed(RunMultiParallelExp)(var_par,Param) for var_par in variables)
        