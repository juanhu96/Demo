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
from utils.import_parameters import import_basics, import_estimation, import_MNL_estimation


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
    
    # a = np.array([[0.5, 0.2, 0.1], [0.2, 0.8, 0]])
    # print(a)
    # V_total = a / (1-a)
    # print(V_total)

    # v_total = V_total.flatten()
    # print(v_total)
    # V_temp = v_total.reshape((2, 3))
    # V_nonzero = np.where(V_temp == 0, np.inf, V_temp)
    # v_min = np.min(V_nonzero, axis=1)
    # big_M = np.where(v_min != 0, 1/v_min, 0)
    # print(f'Big M is {big_M} The min for correponding big M is {np.min(big_M)} and max is {np.max(big_M)}\n')
    # gamma = (v_total * v_total) / (1 + v_total)
    # return 



    Facility_BLP_models = ['MaxVaxHPIDistBLP']
    Facility_LogLin_models = ['MaxVaxDistLogLin']
    Assortment_MNL_models = ['MNL', 'MNL_partial', 'MNL_partial_new']
    test_models = ['MNL_partial_new_test']

    # test(Chain, R, A, setting_tag)

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

    if Model in test_models:
        F_current, F_total = import_estimation('BLP_matrix', Chain, R, A, None, setting_tag) # willingness
        V_total = F_total / (1-F_total)
        

        
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
    

    if Model in Assortment_MNL_models or Model in test_models:

        v_total = V_total.flatten()
        # test_new(Chain, R, A, v_total, p_total, Closest_total, num_tracts, num_current_stores, num_total_stores, setting_tag)
        pv_total = p_total * v_total

        v_total = v_total * Closest_total
        pv_total = pv_total * Closest_total # make sure zero at other place

        ### M for MNL_partial_new
        if Model == 'MNL_partial_new' or Model == 'MNL_partial_new_test':
            
            # test_new(Chain, R, A, v_total, p_total, Closest_total, num_tracts, num_current_stores, num_total_stores, setting_tag)
            # return
        
            V_temp = v_total.reshape((num_tracts, num_total_stores))
            V_nonzero = np.where(V_temp == 0, np.inf, V_temp)
            # V_nonzero = np.where(V_temp == 0, 1000, V_temp) # np.inf results in v_min = 0, then big_M = 0
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

    elif Model == 'MNL_partial_new' or Model == 'MNL_partial_new_test':

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


def test_new(Chain, R, A, v_total, p_total, Closest_total, num_tracts, num_current_stores, num_total_stores, setting_tag):

    V_temp = v_total.reshape((num_tracts, num_total_stores))

    sorted_indices_v_total = np.argsort(V_temp.flatten())[-3:][::-1]
    largest_values_v_total = V_temp.flatten()[sorted_indices_v_total]
    print("\nThe three largest values in V_total are:")
    for index, value in zip(sorted_indices_v_total, largest_values_v_total):
        print(f"Value: {value}, Index: {np.unravel_index(index, V_temp.shape)}")

    zero_rows = np.where(np.all(V_temp == 0, axis=1))[0]
    print("Indices of rows that are all zeros:", zero_rows) # tons of rows with all zero

    # there are rows with all V_temp == 0
    V_nonzero = np.where(V_temp == 0, np.inf, V_temp)
    v_min = np.min(V_nonzero, axis=1)
    big_M = np.where(v_min != 0, 1/v_min, 0)
    print(f'The min for correponding big M is {np.min(big_M)} and max is {np.max(big_M)}\n')
    gamma = (v_total * v_total) / (1 + v_total)
    pg_total = p_total * gamma
    pg_total = pg_total * Closest_total
    
    return


def test(Chain, R, A, setting_tag):

    F_current, F_total = import_estimation('BLP_matrix', Chain, R, A, None, setting_tag) # willingness, p

    U_current = F_current / (1-F_current)
    U_total = F_total / (1-F_total)
    V_current, V_total = import_estimation('V', Chain, R, A, None, setting_tag)

    print(f'The max of F / 1-F current and total is {np.amax(U_current)} and {np.amax(U_total)}\n')
    print(f'The max of wrong V is {np.amax(V_current)} and {np.amax(V_total)}\n')

    # ========================================================================
    # For U_current
    sorted_indices_u_current = np.argsort(U_current.flatten())[-3:][::-1]
    largest_values_u_current = U_current.flatten()[sorted_indices_u_current]
    print("The three largest values in U_current are:")
    for index, value in zip(sorted_indices_u_current, largest_values_u_current):
        print(f"Value: {value}, Index: {np.unravel_index(index, U_current.shape)}")

    # For U_total
    sorted_indices_u_total = np.argsort(U_total.flatten())[-3:][::-1]
    largest_values_u_total = U_total.flatten()[sorted_indices_u_total]
    print("\nThe three largest values in U_total are:")
    for index, value in zip(sorted_indices_u_total, largest_values_u_total):
        print(f"Value: {value}, Index: {np.unravel_index(index, U_total.shape)}")

    # For V_current
    sorted_indices_v_current = np.argsort(V_current.flatten())[-3:][::-1]
    largest_values_v_current = V_current.flatten()[sorted_indices_v_current]
    print("\nThe three largest values in V_current are:")
    for index, value in zip(sorted_indices_v_current, largest_values_v_current):
        print(f"Value: {value}, Index: {np.unravel_index(index, V_current.shape)}")

    # For V_total
    sorted_indices_v_total = np.argsort(V_total.flatten())[-3:][::-1]
    largest_values_v_total = V_total.flatten()[sorted_indices_v_total]
    print("\nThe three largest values in V_total are:")
    for index, value in zip(sorted_indices_v_total, largest_values_v_total):
        print(f"Value: {value}, Index: {np.unravel_index(index, V_total.shape)}")

    # NOTE: U_current matches V_current, U_total (6400, 6415, 5901) does not matches V_total (4892, 6400, 6415)

    # ========================================================================

    print(U_total[5901, 4745], V_total[5901, 4745]) # 22.87026397602321 22.945411822248474
    print(U_total[4892, 4988], V_total[4892, 4988]) # 22.458366224000272 24.33190106928219


    V_temp = v_total.reshape((num_tracts, num_total_stores))
    V_nonzero = np.where(V_temp == 0, np.inf, V_temp)
    v_min = np.min(V_nonzero, axis=1)
    big_M = np.where(v_min != 0, 1/v_min, 0)
    print(f'The min for correponding big M is {np.min(big_M)} and max is {np.max(big_M)}\n')
    gamma = (v_total * v_total) / (1 + v_total)
    pg_total = p_total * gamma
    pg_total = pg_total * Closest_total
    
    return

    from sklearn.metrics.pairwise import cosine_similarity
    vec1 = U_current.flatten().reshape(1, -1)
    vec2 = V_current.flatten().reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    print(f"Cosine similarity: {similarity}")

    print(np.amax(vec1 - vec2))

    return 