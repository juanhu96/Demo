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

scale_factor = 10000


def optimize_chain(Chain_type, Model, M, K, expdirpath, constraint_list = ['assigned', 'vaccinated']):

    print(f'Chain type: {Chain_type}; Model: {Model}; M = {str(M)}, K = {str(K)}. Results stored at {expdirpath}')
    
    Population, Quartile, C_current, C_chains, C_total, Closest_current, Closest_total, C_currentMinDist, C_totalMinDist, num_tracts, num_current_stores, num_chains_stores, num_total_stores = import_dist(Chain_type, M)

    # TODO: do matching between tract & zip, and also include demographics
    
    BLP_models = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP']
    LogLin_models = ['MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin']
    
    if Model in BLP_models: 
        #Demand_parameter = [[1.227, -0.452], [2.028, -0.120, -1.357, -1.197, -0.925, -0.254, -0.218, -0.114]] # v3
        Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]] # v2
        F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_BLP(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)

    if Model in LogLin_models: 
        Demand_parameter = [[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, 0.053, 0.047, 0.039]]
        F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)

    ###########################################################################
    
    # n copies of demand
    p_total = np.tile(Population, num_total_stores)
    p_total = np.reshape(p_total, (num_total_stores, num_tracts))
    p_total = p_total.T.flatten()
       
    p_current = np.tile(Population, num_current_stores)
    p_current = np.reshape(p_current, (num_current_stores, num_tracts))
    p_current = p_current.T.flatten()
    
    # travel distance
    c_current = C_current.flatten() / scale_factor
    c_total = C_total.flatten() / scale_factor
    
    # population * distance 
    pc_current = p_current * c_current
    pc_total = p_total * c_total


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

    
    ###########################################################################

    if Model == 'MaxVaxHPIDistBLP' or Model == 'MaxVaxHPIDistLogLin':

        for constraint in constraint_list:

            if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')
            print(expdirpath + constraint + '/')

            if Chain_type == 'Dollar':
                optimize_rate(scenario='current', constraint=constraint,
                            pc=pc_current, 
                            pf=pfdh_current, 
                            ncp=p_current, p=Population,
                            closest=Closest_current, K=K, 
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores, 
                            num_tracts=num_tracts,
                            scale_factor=scale_factor,
                            path = expdirpath + constraint + '/')

            optimize_rate(scenario='total', constraint=constraint,
                        pc=pc_total,
                        pf=pfdh_total,
                        ncp=p_total, p=Population, 
                        closest=Closest_total, K=K,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath + constraint + '/')

    ###########################################################################
    
    if Model == 'MaxVaxDistBLP' or Model == 'MaxVaxDistLogLin':
        
        for constraint in constraint_list:

            if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')
            print(expdirpath + constraint + '/')

            if Chain_type == 'Dollar':
                optimize_rate(scenario = 'current', constraint=constraint,
                            pc=pc_current, 
                            pf=pfd_current, 
                            ncp=p_current, p=Population,
                            closest=Closest_current, K=K, 
                            num_current_stores=num_current_stores, 
                            num_total_stores=num_total_stores, 
                            num_tracts=num_tracts,
                            scale_factor=scale_factor, 
                            path = expdirpath + constraint + '/')
                
            optimize_rate(scenario = 'total', constraint = constraint,
                        pc = pc_total, 
                        pf = pfd_total, 
                        ncp = p_total, p = Population,
                        closest = Closest_total, K=K,
                        num_current_stores=num_current_stores, 
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath + constraint + '/')
        
  
    ###########################################################################

    if Model == 'MinDist':

        c_currentMinDist = C_currentMinDist.flatten() / scale_factor
        c_totalMinDist = C_totalMinDist.flatten() / scale_factor

        pc_currentMinDist = p_current * c_currentMinDist
        pc_totalMinDist = p_total * c_totalMinDist

        print(expdirpath)

        if Chain_type == 'Dollar':
            optimize_dist(scenario = 'current',
                        pc = pc_currentMinDist,  ncp = p_current, p = Population, K=K, 
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
        
    ###########################################################################
    
    if Model == 'MaxVaxFixV':
        
        # population * fix willingness/vaccination rate
        fix_vac_rate = 0.7
        pv_current = p_current * fix_vac_rate
        pv_total = p_total * fix_vac_rate

        for constraint in constraint_list:

            if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')
            print(expdirpath + constraint + '/')

            if Chain_type == 'Dollar':

                optimize_rate_fix(scenario = 'current', constraint = constraint,
                                ncp = p_current, pv = pv_current, p = Population,
                                closest = Closest_current, K=K, 
                                num_current_stores=num_current_stores,
                                num_total_stores=num_total_stores,
                                num_tracts=num_tracts, 
                                scale_factor=scale_factor,
                                path = expdirpath + constraint + '/')

            optimize_rate_fix(scenario = 'total', constraint = constraint,
                            ncp = p_total, pv = pv_total, p = Population, 
                            closest = Closest_total, K=K,
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores, 
                            num_tracts=num_tracts, 
                            scale_factor=scale_factor, 
                            path = expdirpath + constraint + '/')  


