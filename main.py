#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
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
R = int(sys.argv[5]) # R


from utils.optimize_main import optimize_main
from utils.evaluate_main import evaluate_main
from utils.partnership_summary import partnerships_summary
from utils.import_demand import initial_BLP_estimation, import_BLP_estimation, demand_check

def main(Model, Chain, K, M, R):

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

    optimize_main(Model, Chain, M, K, R)
    evaluate_main(Model, Chain, M, K, R)


def initialization(Chain, K):

    initial_BLP_estimation(Chain, K) # with HPI
    # demand_check(Chain_type='Coffee', capacity=8000, heterogeneity=True)



if __name__ == "__main__":
    main(Model, Chain, K, M, R)