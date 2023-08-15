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

from utils.optimize_main import optimize_main
from utils.evaluate_main import evaluate_main
from utils.partnership_summary import partnerships_summary

def main():

    '''
    Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist']
    Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries']
    K_list = [8000, 10000, 12000]
    M_list = [5, 10]
    Demand_estimation = 'BLP'
    
    '''
    

    # optimize_main(Model_list = ['MaxVaxHPIDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # evaluate_main(Model_list = ['MaxVaxHPIDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # partnerships_summary(Model_list = ['MaxVaxHPIDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5], filename='MaxVaxHPIDistBLP')

    # optimize_main(Model_list = ['MaxVaxDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # evaluate_main(Model_list = ['MaxVaxDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # partnerships_summary(Model_list = ['MaxVaxDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5], filename='MaxVaxDistBLP')

    # optimize_main(Model_list = ['MinDist'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # evaluate_main(Model_list = ['MinDist'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])
    # partnerships_summary(Model_list = ['MinDist'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5], filename='MinDist')


    optimize_main(Model_list = ['MaxVaxHPIDistBLP'], Chain_list = ['Dollar'], K_list = [10000], M_list = [5])


if __name__ == "__main__":
    main()