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

from utils.optimize_chain import optimize_chain


def optimize_main(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist'],
                  Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                  K_list = [8000, 10000, 12000],
                  M_list = [5, 10]):


    '''
    Parameters
    ----------

    Model_list: List of strings
        List of models to run experiments

    Chain_list: List of strings
        List of partnerships to consider

    K_list: List of int
        List of capacity to consider


    M_list: List of int
        List of M to consider
    
    '''   


    for Model in Model_list: 
        
        model_path = f'/export/storage_covidvaccine/Result/{Model}/'
        if not os.path.exists(model_path): os.mkdir(model_path)
        
        for K in K_list:
            for M in M_list:

                if M == 5:

                    parameter_path = f'{model_path}M{str(M)}_K{str(K)}/'
                    if not os.path.exists(parameter_path): os.mkdir(parameter_path)

                    for Chain_type in Chain_list:
                        
                        chain_path = f'{parameter_path}{Chain_type}/'
                        if not os.path.exists(chain_path): os.mkdir(chain_path)

                        optimize_chain(Chain_type, Model, M = M, K = K, expdirpath = chain_path)


                elif M == 10 and Model != 'MinDist':
                    
                    parameter_path = f'{model_path}M{str(M)}_K{str(K)}/'
                    if not os.path.exists(parameter_path): os.mkdir(parameter_path)
                        
                    for Chain_type in Chain_list:
                        
                        chain_path = f'{parameter_path}{Chain_type}/'
                        if not os.path.exists(chain_path): os.mkdir(chain_path)
                        
                        optimize_chain(Chain_type, Model, M = M, K = K, expdirpath = chain_path)
    

    
    print('All problems solved!')










