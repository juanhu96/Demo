#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@author: Jingyuan Hu
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys 
Model = sys.argv[1] # Model
Chain = sys.argv[2] # Chain
K = int(sys.argv[3]) # K
M = int(sys.argv[4]) # M
nsplits = int(sys.argv[5])
capcoef = sys.argv[6]
R = sys.argv[7] # R

from utils.optimize_main import optimize_main
from utils.evaluate_main import evaluate_main


def main(Model, Chain, K, M, nsplits, capcoef, R=None, heuristic=True):

    '''
    BLP estimation:
    Capacity_list = [8000, 10000, 12000, 15000, 20000]

    Optimize, Evaluate, Partnership:
    Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist']
    Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries']
    K_list = [8000, 10000, 12000]
    M_list = [5, 10]
    Demand_estimation = 'BLP'
    
    '''

    # optimize_main(Model, Chain, M, K, nsplits, capcoef, R)
    # evaluate_main(Model, Chain, M, K, nsplits, capcoef, R)
    
    if heuristic: 
        assert Model == "MaxVaxHPIDistBLP", "heuristics for BLP only" # only do this for BLP
        optimize_main(Model, Chain, M, K, nsplits, capcoef, R, heuristic=heuristic)
        evalute_main(Model, Chain, M, K, nsplits, capcoef, R, heuristic=heuristic)



if __name__ == "__main__":
    
    if R != 'None':
        main(Model, Chain, K, M, nsplits, capcoef, int(R))
    else:
        main(Model, Chain, K, M, nsplits, capcoef)


