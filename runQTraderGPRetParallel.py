# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:21:39 2020

@author: aless
"""

# 0. importing section initialize logger.--------------------------------------
import logging, sys, os
from utils.readYaml import readConfigYaml
from utils.QLearningGPRet import QTraderObject#,QLinTraderObject


import multiprocessing
from joblib import Parallel, delayed
# from tqdm import tqdm
# from utils.RunExpParallel import RunExp

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

num_cores = multiprocessing.cpu_count()

# 1. Read config ------------------------------------------------------------- 
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramGPRetVarying.yaml'))
logging.info('Successfully read config file with parameters...')

var = Param[Param['varying_par']]

def RunExp(n,Param):
    
    # add varying parameter control
    Param[Param['varying_par']] = n

    if Param['algo'] == 'Q':
        # 2. Instantiate Q Learning Trader 
        QTrader = QTraderObject(Param)
        logging.info('QTrader initialized successfully...')
        #pdb.set_trace()
        # 2. Run experiment ---------------------------------------------------------------- 
        if Param['trainable'] == 1:
            logging.info('Running experiment ex novo...')
            QTrader.TrainQTrader()
            logging.info('Experiment carried out successfully...')
        else:
            logging.info('Running experiment ex post...')
            QTrader.TestFixedQ()
            logging.info('Experiment carried out successfully...')
        
    elif Param['algo'] == 'Qlin': # TODO: finish to implement this part
        # 2. Instantiate Q Learning Trader with linear approximation
        QTrader = QLinTraderObject(Param)
        logging.info('QTrader initialized successfully...')
            
        # 2. Run experiment ---------------------------------------------------------------- 
        QTrader.TrainQTrader()
        logging.info('Experiment carried out successfully...')


if __name__ == "__main__":
    Parallel(n_jobs=num_cores)(delayed(RunExp)(n,Param) for n in var)