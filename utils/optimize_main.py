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

from utils.optimize_model import optimize_rate, optimize_dist, optimize_rate_fix, optimize_rate_MNL, optimize_rate_MNL_partial, optimize_rate_MNL_partial_test, optimize_rate_MNL_partial_new
from utils.import_parameters import import_basics, import_estimation


def optimize_main(Model: str,
                  Chain: str,
                  K: int,
                  M: int,
                  nsplits: int,
                  capcoef: bool,
                  mnl: bool,
                  flexible_consideration: bool,
                  flex_thresh: dict,
                  logdist_above: bool,
                  logdist_above_thresh: float,
                  R, 
                  A,
                  norandomterm: bool,
                  loglintemp: bool,
                  random_seed,
                  setting_tag: str,
                  resultdir='/export/storage_covidvaccine/Result'):
    

    def create_path(Model, Chain, K, M, nsplits, resultdir):
        
        model_path = f'{resultdir}/{Model}'
        if not os.path.exists(model_path): os.mkdir(model_path)
        parameter_path = f'{model_path}/M{str(M)}_K{str(K)}_{nsplits}q'
        if not os.path.exists(parameter_path): os.mkdir(parameter_path)
        chain_path = f'{parameter_path}/{Chain}/'
        if not os.path.exists(chain_path): os.mkdir(chain_path)
        return chain_path
    
    expdirpath = create_path(Model, Chain, K, M, nsplits, resultdir)
    optimize_chain(Model, Chain, K, M, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, loglintemp, setting_tag, expdirpath)

    return



def optimize_chain(Model: str,
                   Chain: str,
                   K: int,
                   M: int,
                   nsplits: int,
                   capcoef: bool,
                   mnl: bool,
                   flexible_consideration: bool,
                   flex_thresh: dict,
                   logdist_above: bool,
                   logdist_above_thresh: float,
                   R, 
                   A,
                   norandomterm: bool,
                   loglintemp: bool,
                   setting_tag: str,
                   expdirpath: str,
                   constraint: str = 'vaccinated',
                   scale_factor: int = 10000):

    print(f'Optimization results stored at {expdirpath}...\n')

    Facility_BLP_models = ['MaxVaxHPIDistBLP']
    Facility_LogLin_models = ['MaxVaxDistLogLin']
    Assortment_MNL_models = ['MNL', 'MNL_partial', 'MNL_partial_test', 'MNL_partial_new']

    # ================================================================================

    (Population, Quartile, abd, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(Chain, M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)
    

    if Model in Facility_BLP_models: 
        V_current, V_total = import_estimation('BLP_matrix', Chain, R, A, None, setting_tag)

    if Model in Facility_LogLin_models: 
        V_current, V_total = import_estimation('LogLin', Chain, R, A, None, setting_tag)
    
    if Model in Assortment_MNL_models:
        V_current, V_total = import_estimation('V', Chain, R, A, None, setting_tag)


    # ================================================================================


    if Model in Facility_BLP_models or Model in Facility_LogLin_models:

        # willingness vector
        v_current = V_current.flatten()
        v_total = V_total.flatten()
        v_total = np.nan_to_num(v_total)

        # population * willingness vector
        pv_current = p_current * v_current
        pv_total = p_total * v_total
        pv_total = pv_total * Closest_total
    

    if Model in Assortment_MNL_models:

        v_total = V_total.flatten()
        pv_total = p_total * v_total

        v_total = v_total * Closest_total
        pv_total = pv_total * Closest_total # make sure zero at other place

        ### M for MNL_partial_new
        if Model == 'MNL_partial_new':
            V_temp = v_total.reshape((num_tracts, num_total_stores))
            V_nonzero = np.where(V_temp == 0, np.inf, V_temp)
            v_min = np.min(V_nonzero, axis=1)
            big_M = np.where(v_min != 0, 1/v_min, 0)
            print(f'The min for correponding big M is {np.min(big_M)} and max is {np.max(big_M)}\n')
            gamma = (v_total * v_total) / (1 + v_total)
            pg_total = p_total * gamma
            pg_total = pg_total * Closest_total


    # ================================================================================


    path = expdirpath + constraint + '/'
    if not os.path.exists(path): os.mkdir(path)

    if Model == 'MaxVaxHPIDistBLP' or Model == 'MaxVaxDistLogLin':

        optimize_rate(scenario='total',
                      constraint=constraint,
                      pv=pv_total,
                      p=Population, 
                      closest=Closest_total,
                      K=K,
                      R=R,
                      A=A,
                      num_current_stores=num_current_stores,
                      num_total_stores=num_total_stores,
                      num_tracts=num_tracts,
                      path=path,
                      setting_tag=setting_tag,
                      scale_factor=scale_factor)

    elif Model == 'MNL_partial':

        optimize_rate_MNL_partial(scenario='total', 
                                  pv=pv_total,
                                  v=v_total,
                                  C=C,
                                  closest=Closest_total,
                                  K=K,
                                  R=R,
                                  A=A,
                                  num_current_stores=num_current_stores,
                                  num_total_stores=num_total_stores,
                                  num_tracts=num_tracts,
                                  path=path,
                                  setting_tag=setting_tag,
                                  scale_factor=scale_factor,
                                  MIPGap=1e-2)

    elif Model == 'MNL_partial_new':

        optimize_rate_MNL_partial_new(scenario='total', 
                                  pg=pg_total, # p * gamma
                                  v=v_total,
                                  C=C,
                                  closest=Closest_total,
                                  K=K,
                                  big_M=big_M,
                                  R=R,
                                  A=A,
                                  num_current_stores=num_current_stores,
                                  num_total_stores=num_total_stores,
                                  num_tracts=num_tracts,
                                  path=path,
                                  setting_tag=setting_tag,
                                  scale_factor=scale_factor,
                                  MIPGap=1e-2)

    else:
        raise Exception("Model undefined")

    # ================================================================================

    # if Model == 'MNL':

    #     optimize_rate_MNL(scenario='total', 
    #                     pf=pf_total,
    #                     v=v_total,
    #                     C=C,
    #                     K=K,
    #                     R=R,
    #                     A=A,
    #                     closest=Closest_total,
    #                     num_current_stores=num_current_stores,
    #                     num_total_stores=num_total_stores,
    #                     num_tracts=num_tracts,
    #                     scale_factor=scale_factor,
    #                     setting_tag=setting_tag,
    #                     path=path,
    #                     MIPGap=1e-2)

    # if Model == 'MinDist':

    #     pc_currentMinDist = p_current * c_currentMinDist
    #     pc_totalMinDist = p_total * c_totalMinDist
       
    #     optimize_dist(scenario = 'total',
    #                 pc = pc_totalMinDist, ncp = p_total, p = Population, K=K,
    #                 num_current_stores=num_current_stores,
    #                 num_total_stores=num_total_stores, 
    #                 num_tracts=num_tracts, 
    #                 scale_factor=scale_factor, 
    #                 path = expdirpath)   
        
    # if Model == 'MaxVaxFixV':
        
    #     # population * fix willingness/vaccination rate
    #     fix_vac_rate = 0.7
    #     pv_current = p_current * fix_vac_rate
    #     pv_total = p_total * fix_vac_rate

    #     if not os.path.exists(expdirpath + constraint + '/'): os.mkdir(expdirpath + constraint + '/')

    #     optimize_rate_fix(scenario = 'total', constraint = constraint,
    #                     ncp = p_total, pv = pv_total, p = Population, 
    #                     closest = Closest_total, K=K,
    #                     num_current_stores=num_current_stores,
    #                     num_total_stores=num_total_stores, 
    #                     num_tracts=num_tracts, 
    #                     scale_factor=scale_factor, 
    #                     path = expdirpath + constraint + '/',
    #                     R = R) 

    return 



