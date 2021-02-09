# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:44:16 2021

@author: alessiobrini
"""

import numpy as np
from typing import Union

def scale_action(action_limit, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = -action_limit ,action_limit
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_limit, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = -action_limit ,action_limit
    return low + (0.5 * (scaled_action + 1.0) * (high - low))

def boltzmann(x: np.ndarray, T: Union[float or int]) -> np.ndarray:
    """
    Get array of Q values and compute the boltmann equation on it

    Parameters
    ----------
    x: np.ndarray
        Array of Q values

    T: float or int
        Temperature parameter. The greater is the more equal different actions are treated

    Returns
    ----------
    y: np.ndarray
        Array of Q values transformed by the boltmann equation

    """
    e_x = np.exp((x - np.max(x))/T)
    y = e_x / e_x.sum(axis=1).reshape(-1,1)
    
    return y