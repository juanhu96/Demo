#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 21, 2022
@author: Jingyuan Hu
"""

import os
os.chdir('/export/storage_covidvaccine/Code')
from utils.optimize_main import optimize_main
from utils.evaluate_main import evaluate_main


def main():

    '''
    Model_list = ['MaxRateHPIDist', 'MaxRateDist', 'MaxRateFixV', 'MinDist']
    Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries']
    K_list = [8000, 10000, 12000]
    Demand_estimation = 'BLP'
    '''
    
    # optimize_main(Model_list = ['MaxRateHPIDist', 'MaxRateDist'], Chain_list = ['Dollar'], Demand_estimation = 'BLP')
    # optimize_main(Model_list = ['MaxRateHPIDist'], Chain_list = ['Dollar'], K_list = [10000], Demand_estimation = 'linear')
    # optimize_main(Model_list = ['MaxRateDist'], Chain_list = ['Dollar'], K_list = [10000], Demand_estimation = 'BLP')

    evaluate_main(Model_list = ['MaxRateDist'], Chain_list = ['Dollar'], K_list = [10000])


if __name__ == "__main__":
    main()