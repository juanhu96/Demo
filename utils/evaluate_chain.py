#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import pandas as pd
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.evaluate_model import evaluate_rate
from utils.construct_F import construct_F_BLP, construct_F_LogLin
from utils.import_dist import import_dist

scale_factor = 10000


def evaluate_chain(Chain_type, Model, M, K, Demand_parameter, expdirpath, constraint_list = ['assigned', 'vaccinated']):
    
    print(f'Evaluating Chain type: {Chain_type}; Model: {Model}; M = {str(M)}, K = {str(K)}. Results stored at {expdirpath}')
    
    Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type, M)
    
    F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_BLP(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)
    
    f_dh_current = F_DH_current.flatten()
    f_dh_total = F_DH_total.flatten()
    pfdh_current = p_current * f_dh_current
    pfdh_total = p_total * f_dh_total

    # Import optimal z from optimziation
    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

        for opt_constr in constraint_list:

            print(f'{expdirpath}{opt_constr}/z...')
            z_total = np.genfromtxt(f'{expdirpath}{opt_constr}/z_total.csv', delimiter = ",", dtype = float)
            z_current = np.genfromtxt(f'{expdirpath}{opt_constr}/z_current.csv', delimiter = ",", dtype = float)

            for eval_constr in constraint_list:

                if Chain_type == 'Dollar':
                    evaluate_rate(scenario = 'current', constraint = eval_constr, z = z_current,
                                pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                                closest = Closest_current, K=K, 
                                num_current_stores=num_current_stores,
                                num_total_stores=num_total_stores, 
                                num_tracts=num_tracts,
                                scale_factor=scale_factor,
                                path = expdirpath + opt_constr + '/')

                evaluate_rate(scenario = 'total', constraint = eval_constr, z = z_total,
                            pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                            closest = Closest_total, K=K,
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores,
                            num_tracts=num_tracts,
                            scale_factor=scale_factor,
                            path = expdirpath + opt_constr + '/')


    else: # MinDist

        print(f'{expdirpath}z...')
        z_total = np.genfromtxt(f'{expdirpath}z_total.csv', delimiter = ",", dtype = float)
        z_current = np.genfromtxt(f'{expdirpath}z_current.csv', delimiter = ",", dtype = float)
        
        constraint_list = ['assigned'] # TEMP
        for eval_constr in constraint_list:

            if Chain_type == 'Dollar':
                evaluate_rate(scenario = 'current', constraint = eval_constr, z = z_current,
                            pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                            closest = Closest_current, K=K, 
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores, 
                            num_tracts=num_tracts,
                            scale_factor=scale_factor,
                            path = expdirpath, 
                            MIPGap = 1e-2)

            evaluate_rate(scenario = 'total', constraint = eval_constr, z = z_total,
                        pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                        closest = Closest_total, K=K,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath,
                        MIPGap = 5e-2)



    pass




