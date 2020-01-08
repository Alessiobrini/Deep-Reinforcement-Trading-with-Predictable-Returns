# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:45:22 2019

@author: aless
"""

# 0. importing section initialize logger.--------------------------------------
import logging, sys, os
from utils.readYaml import readConfigYaml
from utils.QLearningGP import QTraderObject#,QLinTraderObject
import numpy as np

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# 1. Read config ------------------------------------------------------------- 
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramGP.yaml'))
logging.info('Successfully read config file with parameters...')

if Param['algo'] == 'Q':
    # 2. Instantiate Q Learning Trader 
    QTrader = QTraderObject(Param)
    logging.info('QTrader initialized successfully...')
    
    # 2. Run experiment ---------------------------------------------------------------- 
    QTrader.TrainTestQTrader()
    logging.info('Experiment carried out successfully...')
    
elif Param['algo'] == 'Qlin': # TODO: finish to implement this part
    # 2. Instantiate Q Learning Trader with linear approximation
    QTrader = QLinTraderObject(Param)
    logging.info('QTrader initialized successfully...')
    
    argmax, fmax = QTrader.argmaxLinFunc((10,0),np.append(1.0, np.random.rand(len(QTrader.ParamSpace))))
    print(argmax)
    print(fmax)
    
    # 2. Run experiment ---------------------------------------------------------------- 
    QTrader.TrainTestQTrader()
    logging.info('Experiment carried out successfully...')
