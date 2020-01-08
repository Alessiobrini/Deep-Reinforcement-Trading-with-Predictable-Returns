# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:45:22 2019

@author: aless
"""

# 0. importing section initialize logger.--------------------------------------
import logging, sys, os
from utils.readYaml import readConfigYaml
from utils.QLearningRitter import QTraderObject
from utils.QLearningLinAppRitter import QLinTraderObject
from utils.format_tousands import format_tousands
import numpy as np

import matplotlib.pyplot as plt
import seaborn 
seaborn.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['savefig.dpi'] = 90
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

import pdb

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
Param = readConfigYaml(os.path.join(os.getcwd(), 'config','paramRitter.yaml'))
logging.info('Successfully read config file with parameters...')

obj = []
cumpnl = []
ticks = Param['LotSize']

for lot in Param['LotSize']:
    
    Param['LotSize'] = lot

    if Param['algo'] == 'Q':
        # 2. Instantiate Q Learning Trader 
        QTrader = QTraderObject(Param)
        logging.info('QTrader initialized successfully...')
        # 2. Run experiment ---------------------------------------------------------------- 
        res_df = QTrader.TrainTestQTrader()
        obj.append(res_df['reward'].cumsum().iloc[-1])
        cumpnl.append(res_df['pnl'].cumsum().iloc[-1])
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
    
    else:
        print('Choose an existing algorithm!')


fig = plt.figure()
fig.tight_layout()
plt.suptitle('Results by varying LotSize')
     
objective = fig.add_subplot(211)
objective.plot(ticks,obj)
objective.set_xticks(ticks)
objective.set_xlabel('LotSize')
objective.set_ylabel('Reward')


pnl = fig.add_subplot(212)
pnl.plot(ticks,cumpnl)
pnl.set_xticks(ticks)
pnl.set_xlabel('LotSize')
pnl.set_ylabel('Cumulative PnL')

figpath = os.path.join('outputs', Param['outputDir'],'Qlearning_'+ 
                               format_tousands(Param['N_train']) + 
                               '_Results_by_Lotsize.PNG')
        
# save figure
plt.savefig(figpath)