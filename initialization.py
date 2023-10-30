#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct, 2023
@author: Jingyuan Hu
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys 
Chain = sys.argv[1]
K = int(sys.argv[2])
nsplits = int(sys.argv[3])
capcoef = sys.argv[4]

from utils.initial_BLP import initial_BLP_estimation, demand_check


def initialization(Chain, K, nsplits, capcoef, mode='initial'):

    if mode == 'initial': initial_BLP_estimation(Chain, K, nsplits, capcoef)
    elif mode == 'check': demand_check(Chain, K, nsplits, capcoef)
    else: raise Exception("Mode undefined, has to be initial or check\n")

    return




if __name__ == "__main__":
    initialization(Chain, K, nsplits, capcoef)