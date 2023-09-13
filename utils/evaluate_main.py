#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.evaluate_chain import evaluate_chain_RandomFCFS


def evaluate_main(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist'],
                  Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                  K_list = [8000, 10000, 12000],
                  M_list = [5, 10]):
    
    '''
    Evaluate the optimal locations from the optimization model

    Parameters
    ----------

    Model_list: List of strings
        Models to evaluate

    Chain_list: List of strings
       Partnerships to evaluate

    K_list: List of int
        Capacity to evaluate
    
    ''' 

    for Model in Model_list: 
        for K in K_list:
            for M in M_list:
                for Chain_type in Chain_list:
                        
                    chain_path = f'/export/storage_covidvaccine/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'
                    evaluate_chain_RandomFCFS(Chain_type, Model, M = M, K = K, expdirpath = chain_path)



    pass