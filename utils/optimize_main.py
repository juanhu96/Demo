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

from utils.optimize_model import optimize_rate, optimize_dist, optimize_rate_fix, optimize_rate_MNL, optimize_rate_MNL_partial, optimize_rate_MNL_partial_test
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
    Assortment_MNL_models = ['MNL', 'MNL_partial', 'MNL_partial_test']

    # ================================================================================

    (Population, Quartile, abd, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(Chain, M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)


    # model_mapping = {
    # 'Facility_BLP_models': ('BLP_matrix',),
    # 'Facility_LogLin_models': ('LogLin',),
    # 'Assortment_MNL_models': ('V',)
    # }

    # model_key = next((k for k in model_mapping if Model in k), None)
    # if model_key:
    #     args = model_mapping[model_key]
    #     if model_key == 'Assortment_MNL_models':
    #         V_current, V_total = import_estimation(*args, Chain, R, A, setting_tag)
    #     else:
    #         F_current, F_total = import_estimation(*args, Chain, R, A, setting_tag)
    # else:
    #     raise ValueError("Model not recognized")

    if Model in Facility_BLP_models: 
        F_current, F_total = import_estimation('BLP_matrix', Chain, R, A, setting_tag)

    if Model in Facility_LogLin_models: 
        F_current, F_total = import_estimation('LogLin', Chain, R, A, setting_tag)
    
    if Model in Assortment_MNL_models:
        V_current, V_total = import_estimation('V', Chain, R, A, setting_tag)


    # ================================================================================


    if Model in Facility_BLP_models or Model in Facility_LogLin_models:

        # willingness vector
        f_current = F_current.flatten()
        f_total = F_total.flatten()
        f_total = np.nan_to_num(f_total)

        # population * willingness vector
        pf_current = p_current * f_current
        pf_total = p_total * f_total
        pf_total = pf_total * Closest_total
    

    if Model in Assortment_MNL_models:
        v_total = V_total.flatten()
        pf_total = p_total * v_total

        v_total = v_total * Closest_total
        pf_total = pf_total * Closest_total # make sure zero at other place


    # ================================================================================


    path = expdirpath + constraint + '/'
    if not os.path.exists(path): os.mkdir(path)

    if Model == 'MaxVaxHPIDistBLP' or Model == 'MaxVaxDistLogLin':

        optimize_rate(scenario='total',
                      constraint=constraint,
                      pc=pc_total,
                      pf=pf_total,
                      ncp=p_total,
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
                                  pf=pf_total,
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

    elif Model == 'MNL_partial_test':

        raise Exception("Broken code")

        z_file_name = '/export/storage_covidvaccine/Result/MaxVaxDistLogLin/M5_K10000_4q/Dollar/vaccinated/z_total_10000_5_4q_mnl'
        z_total = np.genfromtxt(f'{z_file_name}.csv', delimiter = ",", dtype = float) 
        optimize_rate_MNL_partial_test(z=z_total,
                        scenario='total', 
                        pf=pf_total,
                        v=v_total,
                        C=C,
                        K=K,
                        R=R,
                        closest=Closest_total,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        setting_tag=setting_tag,
                        path=path,
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



