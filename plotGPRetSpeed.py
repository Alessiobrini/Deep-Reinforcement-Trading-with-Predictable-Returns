# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:32:32 2020

@author: aless
"""

# 0. importing section initialize logger.--------------------------------------
import logging, sys, os, pdb
from utils.readYaml import readConfigYaml
from utils.ResultsAnalyzer import ResultsObject

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
Param = readConfigYaml(os.path.join(os.getcwd(),'config','paramGPRetPlotSpeed.yaml'))
logging.info('Successfully read config file with parameters...')

# pdb.set_trace()
# 2. Instantiate Q Learning Trader 
ResObject = ResultsObject(Param)
logging.info('ResultsAnalyzer initialized successfully...')

# 3. Plot speed results---------------------------------------------------------------- 
if Param['kind'] == 'speed':
    ResObject.PlotLearningSpeed()
    logging.info('Speeds plotted successfully...')
elif Param['kind'] == 'asym':
    ResObject.PlotAsymptoticResults()
    logging.info('Asymptotic results plotted successfully...')
else:
    print('Choose an available type of plot!')