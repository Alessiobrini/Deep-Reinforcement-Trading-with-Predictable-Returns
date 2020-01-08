# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:30:06 2019

@author: simone
"""

import yaml

def readConfigYaml(filepath):
    # used to read config file. 
    # This will allow us to pass as input any values to the code without the need
    # of changing any hardcoded values.
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)#,Loader=yaml.FullLoader)
        
    return cfg