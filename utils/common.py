# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:33:01 2021

@author: alessiobrini
"""

import logging, sys
from typing import Union
import os

# yaml is bugged and not able to read float written in scientific notation
# for info look at https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
# https://yaml.readthedocs.io/en/latest/install.html
import ruamel.yaml as yaml
import warnings

warnings.simplefilter("ignore", yaml.error.UnsafeLoaderWarning)


def generate_logger() -> logging.RootLogger:
    """Returns the logger

    Returns
    -------
    root: logging.RootLogger
        the callable logger
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    return root  # todo write the datatype


def format_tousands(num: int) -> str:
    """Takes an integer number and returns a string which express that integer
    in unit of thousands. It is useful to store experiments in folder named by
    their training runtime.

    Parameters
    ----------
    num: int
        The integer number you want to transform

    Returns
    -------
    str_num: str
        The converted number in text format
    """
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    str_num = "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude]
    ).replace(".", "_")

    return str_num


def readConfigYaml(filepath) -> dict:
    """Takes a specified path for the config file and open it. This allows
    to pass all the inputs by modifying the correspondent config file.

    Parameters
    ----------
    filepath: str
        The path where the config file is stored

    Returns
    -------
    cfg: dict
        The opened config file stored as a dict
    """
    with open(filepath, "r") as ymlfile:
        cfg = yaml.load(ymlfile)  # ,Loader=yaml.FullLoader)

    return cfg


def saveConfigYaml(config: dict, path: str, multitest: bool = False):
    """Takes the config and a specified path where to save it. This allows
    to save the config file in the folde rof the experiment to enhance reproducibility.

    Parameters
    ----------
    config: dict
        The dictionary to store
    path: str

    """
    if multitest:
        with open(
            os.path.join(
                path,
                "MultiTestconfig_{}.yaml".format(format_tousands(config["N_test"])),
            ),
            "w",
        ) as file:
            file.write(yaml.dump(config))
    else:
        with open(
            os.path.join(
                path, "config_{}.yaml".format(format_tousands(config["N_train"]))
            ),
            "w",
        ) as file:
            file.write(yaml.dump(config))


def GeneratePathFolder(
    outputDir: str,
    outputClass: str,
    outputModel: str,
    varying_pars: Union[str or None],
    N_train: int,
    Param: dict,
) -> Union[str, bytes, os.PathLike]:

    """
    Create proper directory tree for storing results and returns the experiment path.

    Parameters
    ----------
    outputDir: str
        Main directory for output results

    outputClass: str
        Subdirectory usually indicating the family of algorithms e.g. "DQN"

    outputModel: str
        Subdirectory indicating the name of the experiments

    varying_pars: Union[str or None]
        Name of varying hyperparameters when performing parallel experiments

    N_train: int
        Length of the experiment

    Param: dict
        Dictionary of parameters

    Returns
    -------
    savedpath: str
        The path until the outputModel subdirectory

    """

    # Create directory for outputs
    if varying_pars:
        savedpath = os.path.join(
            os.getcwd(),
            outputDir,
            outputClass,
            outputModel,
            format_tousands(N_train),
            "_".join(
                [
                    str(v) + "_" + "_".join([str(val) for val in Param[v]])
                    if isinstance(Param[v], list)
                    else str(v) + "_" + str(Param[v])
                    for v in Param["varying_pars"]
                ]
            ),
        )

    #'_'.join([str((v,str(Param[v]))) for v in varying_pars]))

    else:
        savedpath = os.path.join(
            os.getcwd(), outputDir, outputClass, outputModel, format_tousands(N_train)
        )

    # use makedirs to create a tree of subdirectory
    if not os.path.exists(savedpath):
        os.makedirs(savedpath)
    else:
        if Param["varying_type"] == "random_search":
            pass
        else:
            sys.exit("Folder already exists. This experiment has already been run.")

    return savedpath

def set_size(width, fraction=1, subplots=(1, 1)) -> tuple:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    # fig_height_in = fig_width_in * golden_ratio
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def prime_factors(n: int) -> list:
    """Compute the decomposition of an integer number into prime factors.

    Parameters
    ----------
    n: int
        Integer number to decompose into its prime factors
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst.
    
    Parameters
    ----------
    lst: list
            List to split into chunks
    n: int, optional
            Number of chunks for partitioning the list

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def save(config, path):
    """Takes the config and a specified path where to save it. This allows
    to save the config file in the folde rof the experiment to enhance reproducibility.

    Parameters
    ----------
    config: dict
        The dictionary to store
    path: str

    """
    with open(os.path.join(path, "paramMultiTestOOS.yaml"), "w") as file:
        file.write(yaml.dump(config))