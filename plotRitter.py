# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:08:38 2019

@author: aless
"""

# 0. importing section initialize logger.--------------------------------------
import logging, sys, os
from utils.readYaml import readConfigYaml
from utils.QLearningRitter import QTraderObject
from utils.generateLogger import generate_logger

logger = generate_logger()

# 1. Read config ------------------------------------------------------------- 
Param = readConfigYaml(os.path.join(os.getcwd(), 'config\\paramRitter.yaml'))
logging.info('Successfully read config file with parameters...')
# 2. Instantiate QTrader ---------------------------------------------------------------- 
QTrader = QTraderObject(Param)
logging.info('QTrader initialized successfully...')

## 2. Plot results ---------------------------------------------------------------- 
# QTrader.plot_Heatmap()
QTrader.plot_QValueFunction()
QTrader.plot_Actions()
logging.info('Experiment results plotted out successfully...')
