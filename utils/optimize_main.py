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

from utils.optimize_model import optimize_rate, optimize_dist, optimize_rate_fix
from utils.construct_F import construct_F_BLP, construct_F_LogLin
from utils.import_dist import import_dist
from utils.import_demand import import_BLP_estimation



def optimize_main(Model, Chain, M, K, groups, capcoef, R=None, constraint='vaccinated', resultdir='/export/storage_covidvaccine/Result'):
        
    model_path = f'{resultdir}/{Model}'
    if not os.path.exists(model_path): os.mkdir(model_path)

    # parameter_path = f'{model_path}/M{str(M)}_K{str(K)}'
    if capcoef: parameter_path = f'{model_path}/M{str(M)}_K{str(K)}_{groups}q_capcoef'
    else: parameter_path = f'{model_path}/M{str(M)}_K{str(K)}_{groups}q'
    if not os.path.exists(parameter_path): os.mkdir(parameter_path)

    chain_path = f'{parameter_path}/{Chain}/'
    if not os.path.exists(chain_path): os.mkdir(chain_path)

    optimize_chain(Model, Chain, M, K, groups, capcoef, chain_path, R, constraint=constraint)

    return




def optimize_chain(Model, Chain, M, K, groups, capcoef, expdirpath, R=None, constraint='vaccinated', scale_factor=10000):

    print(f'Start optimization with {Chain}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}.\n Results stored at {expdirpath}\n')
    
    Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, _, _, num_tracts, num_current_stores, num_total_stores = import_dist(Chain, M)
    
    BLP_models = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP']
    LogLin_models = ['MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin']

    if Model in BLP_models: 
        # TODO: need to utilize
        F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain, K, groups, capcoef) # F_D_current, F_D_total are just dummy

    if Model in LogLin_models: 
        # TODO: import demand parameter from other directories
        Demand_parameter = [[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, 0.053, 0.047, 0.039]] 
        F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)


    # ================================================================================


    if Model in BLP_models or Model in LogLin_models:
        
        # willingness
        f_d_current = F_D_current.flatten()
        f_d_total = F_D_total.flatten()
        
        f_dh_current = F_DH_current.flatten()
        f_dh_total = F_DH_total.flatten()
        
        
        # population * willingness
        pfd_current = p_current * f_d_current
        pfd_total = p_total * f_d_total
        
        pfdh_current = p_current * f_dh_current
        pfdh_total = p_total * f_dh_total

    
    # ================================================================================

    if Model == 'MaxVaxHPIDistBLP' or Model == 'MaxVaxHPIDistLogLin':

        if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

        if Chain == 'Dollar':
            optimize_rate(scenario='current', constraint=constraint,
                        pc=pc_current, 
                        pf=pfdh_current, 
                        ncp=p_current, p=Population,
                        closest=Closest_current, K=K, 
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores, 
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath + constraint + '/',
                        R = R)

        optimize_rate(scenario='total', constraint=constraint,
                    pc=pc_total,
                    pf=pfdh_total,
                    ncp=p_total, p=Population, 
                    closest=Closest_total, K=K,
                    num_current_stores=num_current_stores,
                    num_total_stores=num_total_stores,
                    num_tracts=num_tracts,
                    scale_factor=scale_factor,
                    path = expdirpath + constraint + '/',
                    R = R)

    # ================================================================================
    
    if Model == 'MaxVaxDistBLP' or Model == 'MaxVaxDistLogLin':

        if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

        if Chain == 'Dollar':
            optimize_rate(scenario = 'current', constraint=constraint,
                        pc=pc_current, 
                        pf=pfd_current, 
                        ncp=p_current, p=Population,
                        closest=Closest_current, K=K, 
                        num_current_stores=num_current_stores, 
                        num_total_stores=num_total_stores, 
                        num_tracts=num_tracts,
                        scale_factor=scale_factor, 
                        path = expdirpath + constraint + '/',
                        R = R)
                
        optimize_rate(scenario = 'total', constraint = constraint,
                    pc = pc_total, 
                    pf = pfd_total, 
                    ncp = p_total, p = Population,
                    closest = Closest_total, K=K,
                    num_current_stores=num_current_stores, 
                    num_total_stores=num_total_stores,
                    num_tracts=num_tracts,
                    scale_factor=scale_factor,
                    path = expdirpath + constraint + '/',
                    R = R) 
  
    # ================================================================================

    if Model == 'MinDist':

        pc_currentMinDist = p_current * c_currentMinDist
        pc_totalMinDist = p_total * c_totalMinDist

        if Chain == 'Dollar':
            optimize_dist(scenario = 'current',
                        pc = pc_currentMinDist, ncp = p_current, p = Population, K=K, 
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores, 
                        num_tracts=num_tracts, 
                        scale_factor=scale_factor,
                        path = expdirpath)
       
        optimize_dist(scenario = 'total',
                    pc = pc_totalMinDist, ncp = p_total, p = Population, K=K,
                    num_current_stores=num_current_stores,
                    num_total_stores=num_total_stores, 
                    num_tracts=num_tracts, 
                    scale_factor=scale_factor, 
                    path = expdirpath)   
        
    # ================================================================================
    
    if Model == 'MaxVaxFixV':
        
        # population * fix willingness/vaccination rate
        fix_vac_rate = 0.7
        pv_current = p_current * fix_vac_rate
        pv_total = p_total * fix_vac_rate

        if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

        if Chain == 'Dollar':

            optimize_rate_fix(scenario = 'current', constraint = constraint,
                            ncp = p_current, pv = pv_current, p = Population,
                            closest = Closest_current, K=K, 
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores,
                            num_tracts=num_tracts, 
                            scale_factor=scale_factor,
                            path = expdirpath + constraint + '/',
                            R = R)

        optimize_rate_fix(scenario = 'total', constraint = constraint,
                        ncp = p_total, pv = pv_total, p = Population, 
                        closest = Closest_total, K=K,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores, 
                        num_tracts=num_tracts, 
                        scale_factor=scale_factor, 
                        path = expdirpath + constraint + '/',
                        R = R) 

    return 



