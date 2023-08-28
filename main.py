#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu
"""

import os
import numpy as np
import pandas as pd # temp
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.optimize_main import optimize_main
from utils.evaluate_main import evaluate_main
from utils.partnership_summary import partnerships_summary
from utils.import_demand import initial_BLP_estimation, import_BLP_estimation, demand_check

def main():

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

    ############################################################################################################

    # for capacity in [8000, 12000, 15000, 20000]:
    #     initial_BLP_estimation(Chain_type='Dollar', capacity=capacity)
    # demand_check(Chain_type='Dollar', capacity=100000)

    ############################################################################################################

    # optimize_main(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxHPIDistLogLin', 'MinDist'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # evaluate_main(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxHPIDistLogLin', 'MinDist'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # partnerships_summary(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxHPIDistLogLin', 'MinDist'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5], filename = 'all')

    ############################################################################################################

    

if __name__ == "__main__":
    main()