# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:23:53 2021

@author: alessiobrini
"""

# delete any variables created in previous run if you are using this script on Spyder
import os

if any("SPYDER" in name for name in os.environ):
    from IPython import get_ipython

    get_ipython().magic("reset -sf")
    
import os
import pandas as pd
import numpy as np
import pdb

# go up one level
path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)

data_folder_equity = os.path.join(os.getcwd(), 'data', 'daily_equity')
# take all the names of the parquet files
equity_files = [f for f in os.listdir(data_folder_equity) if f.endswith('gzip')]

# split equity categories
us_equities = [f for f in equity_files if not f.split('daily_bars_')[1].startswith('L.')]
lse_equities = [f for f in equity_files if f.split('daily_bars_')[1].startswith('L.')]

dfs = []

for name in us_equities:
        
    sym = name.split('daily_bars_')[1].split('_')[0]
    

    df = pd.read_parquet(os.path.join(data_folder_equity,name))
    df.set_index('date', inplace=True)
    
    # construct and set multilevel index
    iterables = [[sym], list(df.index)]
    multi_idx = pd.MultiIndex.from_product(iterables, names=["first", "second"])
    df.index = multi_idx
    
    # append to a list
    dfs.append(df)
    
full_df = pd.concat(dfs, axis=0)
full_df.to_parquet(os.path.join(os.getcwd(), 'data', 'us_equities'), compression='gzip')

