# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:36:51 2020

@author: aless
"""

import logging, sys


def generate_logger():
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
    return root


# def generate_logger():
#     import logging
#     LOG_FILENAME = os.path.join(PROJECT_DIR, "mylog.log")
#     FORMAT = "%(asctime)s : %(message)s"
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     # Reset the logger.handlers if it already exists.
#     if logger.handlers:
#         logger.handlers = []
#     fh = logging.FileHandler(LOG_FILENAME)
#     formatter = logging.Formatter(FORMAT)
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#     return logger
