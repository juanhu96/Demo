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

from utils.optimize_model import optimize_rate, optimize_dist, optimize_rate_fix, optimize_rate_MNL
from utils.import_parameters import import_basics, import_BLP_estimation, import_LogLin_estimation, import_MNL_estimation
from utils.heuristic import rescale_estimation



def optimize_main(Model, Chain, M, K, nsplits, capcoef, R=None, constraint='vaccinated', heuristic=False, resultdir='/export/storage_covidvaccine/Result'):
    
    
    # Model
    model_path = f'{resultdir}/{Model}'
    if not os.path.exists(model_path): os.mkdir(model_path)


    # Scenario
    if capcoef: parameter_path = f'{model_path}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef'
    else: parameter_path = f'{model_path}/M{str(M)}_K{str(K)}_{nsplits}q'
    if not os.path.exists(parameter_path): os.mkdir(parameter_path)


    # Chain
    chain_path = f'{parameter_path}/{Chain}/'
    if not os.path.exists(chain_path): os.mkdir(chain_path)


    optimize_chain(Model, Chain, M, K, nsplits, capcoef, chain_path, R, constraint, heuristic)

    return




def optimize_chain(Model, Chain, M, K, nsplits, capcoef, expdirpath, R, constraint, heuristic, scale_factor=10000):

    print(f'Start optimization with Model={Model}, M={M}, K={K}, Chain={Chain}, nsplits={nsplits}, capcoef={capcoef}, R={R}, heuristic={heuristic}. \n Results stored at {expdirpath}\n')
    
    Population, Quartile, abd, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, _, _, C, num_tracts, num_current_stores, num_total_stores = import_basics(Chain, M, nsplits)

    BLP_models = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP']
    LogLin_models = ['MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin']
    MNL_models = ['MNL']

    if Model in BLP_models: 
        F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain, K, nsplits, capcoef) # F_D_current, F_D_total are just dummy

    if Model in LogLin_models: 
        F_D_current, F_D_total, F_DH_current, F_DH_total = import_LogLin_estimation(C_total, Quartile, abd, nsplits, num_tracts, num_current_stores) # F_DH_current, F_DH_total are just dummy
    
    if heuristic:
        F_DH_total = rescale_estimation(F_DH_total, Model, Chain, M, K, nsplits, capcoef, R, constraint)
    
    if Model in MNL_models:
        V_current, V_total = import_MNL_estimation(Chain, K, nsplits, capcoef)

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



    if Model in MNL_models:

        v_total = V_total.flatten()
        pfdh_total = p_total * v_total
        pfdh_total = pfdh_total * Closest_total # make sure v_ij are zero other place
        

    # ================================================================================

    if Model == 'MNL':

        if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

        optimize_rate_MNL(scenario='total', 
                        pf=pfdh_total,
                        v=v_total,
                        C=C,
                        K=K,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        R = R,
                        path=expdirpath + constraint + '/')



    if Model == 'MaxVaxHPIDistBLP':

        if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

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
                    R = R,
                    heuristic=heuristic)

    # ================================================================================
    
    if Model == 'MaxVaxDistLogLin':

        if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

        pfd_total_zero = pfd_total * Closest_total 

        optimize_rate(scenario = 'total', constraint = constraint,
                    pc = pc_total, 
                    pf = pfd_total_zero, 
                    ncp = p_total, p = Population,
                    closest = Closest_total, K=K,
                    num_current_stores=num_current_stores, 
                    num_total_stores=num_total_stores,
                    num_tracts=num_tracts,
                    scale_factor=scale_factor,
                    path = expdirpath + constraint + '/',
                    R = R,
                    MIPGap = 5e-2)
  
    # ================================================================================

    if Model == 'MinDist':

        pc_currentMinDist = p_current * c_currentMinDist
        pc_totalMinDist = p_total * c_totalMinDist
       
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



