# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:13:36 2019

@author: aless
"""


def format_tousands(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude]
    ).replace(".", "_")
