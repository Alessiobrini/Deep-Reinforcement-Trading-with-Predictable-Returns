# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:30:06 2019

@author: simone
"""

# import yaml
# yaml is bugged and not able to read float written in scientific notation
# for info look at https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
# https://yaml.readthedocs.io/en/latest/install.html
import ruamel.yaml as yaml
import os, pdb
import warnings

warnings.simplefilter("ignore", yaml.error.UnsafeLoaderWarning)

from utils.format_tousands import format_tousands


def readConfigYaml(filepath):
    # used to read config file.
    # This will allow us to pass as input any values to the code without the need
    # of changing any hardcoded values.
    with open(filepath, "r") as ymlfile:
        cfg = yaml.load(ymlfile)  # ,Loader=yaml.FullLoader)

    return cfg


def saveConfigYaml(config, path):

    with open(
        os.path.join(path, "config_" + format_tousands(config["N_train"]) + ".yaml"),
        "w",
    ) as file:
        file.write(yaml.dump(config))


def saveMultiTestConfigYaml(config, path):

    with open(
        os.path.join(
            path, "MultiTestconfig_" + format_tousands(config["N_test"]) + ".yaml"
        ),
        "w",
    ) as file:
        file.write(yaml.dump(config))


def save(config, path):

    with open(os.path.join(path, "paramMultiTestOOS.yaml"), "w") as file:
        file.write(yaml.dump(config))
