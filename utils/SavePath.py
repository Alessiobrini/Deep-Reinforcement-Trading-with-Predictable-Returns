# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:40:55 2020

@author: aless
"""
from typing import Tuple, Union
import os, pdb
from utils.format_tousands import format_tousands

def GeneratePathFolder(outputDir: str,
                       outputClass: str,
                       outputModel: str,
                       varying_par: Union[str or NoneType],
                       N_train: int,
                       Param: dict) -> Union[str, bytes, os.PathLike]:
    
    '''
    Create proper directory tree for storing results
    '''
    
    # Create directory for outputs
    if varying_par:

        savedpath = os.path.join(os.getcwd(),
                                 outputDir,
                                 outputClass,
                                 '_'.join([outputModel,varying_par]),
                                 format_tousands(N_train),
                                 '_'.join([varying_par,
                                           str(Param[Param['varying_par']])]))
        
    else:
        savedpath = os.path.join(os.getcwd(),
                                 outputDir,
                                 outputClass,
                                 outputModel,
                                 format_tousands(N_train))
    
    # use makedirs to create a tree of subdirectory
    if not os.path.exists(savedpath):
        os.makedirs(savedpath)
        
    return savedpath