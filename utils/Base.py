# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:36:18 2019

@author: aless
"""

import numpy as np


class Base(object):
    '''
    This class is the core class to implement Q-learning algorithm and its extensions. 
    It is possible to easily extend that framework by incorporating
    different form of transaction costs, objective function and state space representation
    '''

    # ----------------------------------------------------------------------------
    # Init method      
    # ----------------------------------------------------------------------------
    def __init__(self, Param):

        '''
        init method to initialize the class. Parameter inputs are stored 
        as properties of the object.
        '''
        self.Param = Param
        self._setSpaces(Param)
        
    # ----------------------------------------------------------------------------
    # Private method      
    # ----------------------------------------------------------------------------
    def _setSpaces(self, Param):
        '''
        Create discrete action, holding and price spaces
        '''
        ParamSpace = { 
                      'A_space' : np.arange(-Param['K'] ,Param['K']+1, Param['LotSize']),
                      'H_space' : np.arange(-Param['M'],Param['M']+1, Param['LotSize']),
                      'P_space' : np.arange(Param['P_min'],Param['P_max']+1)*Param['TickSize']
                      }

        self.ParamSpace = ParamSpace
    # ----------------------------------------------------------------------------
    # Public method       
    # ----------------------------------------------------------------------------
    
    # ROUNDING FUNCTIONS
    
    def find_nearest_price(self, value):
        '''
        Function to ensure that prices don't exit the valid range
        '''
        array = np.asarray(self.ParamSpace['P_space'])
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    def find_nearest_holding(self, value):
        '''
        Function to ensure that holdings don't exit the valid range
        '''
        array = np.asarray(self.ParamSpace['H_space'])
        idx = (np.abs(array - value)).argmin()
        return array[idx]

