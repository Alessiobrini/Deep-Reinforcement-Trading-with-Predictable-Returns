# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:44:51 2020

@author: aless
"""
import ruamel.yaml as yaml
import os

from utils.format_tousands import format_tousands

def SaveConfig(config, path):
        
    with open(os.path.join(path,format_tousands(config['N_train']) + 
                           '_config.txt'), 'w') as file:
        file.write(yaml.dump(config))
     