# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:40:55 2020

@author: aless
"""
from typing import Tuple, Union
import os, pdb, sys
from utils.format_tousands import format_tousands

def GeneratePathFolder(outputDir: str,
                       outputClass: str,
                       outputModel: str,
                       varying_pars: Union[str or None],
                       N_train: int,
                       Param: dict) -> Union[str, bytes, os.PathLike]:
    
    '''
    Create proper directory tree for storing results
    '''
    
    # Create directory for outputs
    if varying_pars:
        savedpath = os.path.join(os.getcwd(),
                                 outputDir,
                                 outputClass,
                                 outputModel,
                                 format_tousands(N_train),
                                 '_'.join([str(v)+'_'+'_'.join([str(val) for val in Param[v]]) \
                                           if isinstance(Param[v],list) else str(v)+'_' + str(Param[v]) \
                                           for v in Param['varying_pars']]))

#'_'.join([str((v,str(Param[v]))) for v in varying_pars]))
        
    else:
        savedpath = os.path.join(os.getcwd(),
                                 outputDir,
                                 outputClass,
                                 outputModel,
                                 format_tousands(N_train))
    
    # use makedirs to create a tree of subdirectory
    if not os.path.exists(savedpath):
        os.makedirs(savedpath)
    else:
        sys.exit('Folder already exists. This experiment has already been run.')
        
    return savedpath