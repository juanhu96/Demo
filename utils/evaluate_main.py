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

from utils.evaluate_chain import evaluate_chain


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


    # TODO: UPDATE THE COEF

    Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]] # v2
    # Demand_parameter = [[1.227, -0.452], [2.028, -0.120, -1.357, -1.197, -0.925, -0.254, -0.218, -0.114]] # v3


    for Model in Model_list: 
        for K in K_list:
            for M in M_list:

                if M == 5:

                    for Chain_type in Chain_list:
                        
                        chain_path = f'/export/storage_covidvaccine/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'

                        evaluate_chain(Chain_type, Model, M = M, K = K, Demand_parameter = Demand_parameter, expdirpath = chain_path)
                
                elif M == 10 and Model != 'MinDist':

                    for Chain_type in Chain_list:
                        
                        chain_path = f'/export/storage_covidvaccine/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'

                        evaluate_chain(Chain_type, Model, M = M, K = K, Demand_parameter = Demand_parameter, expdirpath = chain_path)


    pass