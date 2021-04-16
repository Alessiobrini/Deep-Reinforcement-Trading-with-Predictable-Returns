import logging, sys, pdb
from typing import Union
from pathlib import Path
import os
import gin
import itertools
import numpy as np


def get_parallelized_combinations(varying_type: str):

    variables = []
    lst = [gin.query_parameter(v) for v in gin.query_parameter("%VARYING_PARS")]
    if varying_type == "combination":
        for xs in itertools.product(*lst):
            variables.append(xs)
    elif varying_type == "ordered_combination":
        for xs in zip(*lst):
            variables.append(xs)
    elif varying_type == "random_search":
        for xs in itertools.product(*lst):
            variables.append(xs)
        variables = [
            variables[i]
            for i in np.random.randint(
                0, len(variables) - 1, gin.query_parameter("%NUM_CORES")
            )
        ]
    elif varying_type == "chunk":
        for xs in itertools.product(*lst):
            variables.append(xs)

    else:
        print("Choose proper way to combine varying parameters")
        sys.exit()

    num_cores = len(variables)

    return variables, num_cores
