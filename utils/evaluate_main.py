#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import numpy as np
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.evaluate_model import compute_distdf, construct_blocks, run_assignment, evaluate_rate, evaluate_rate_MNL_partial
from utils.import_parameters import import_basics, import_BLP_estimation, import_MNL_estimation



def evaluate_main(Model: str,
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
                  RandomFCFS: bool = False,
                  MIP: bool = False,
                  constraint='vaccinated',
                  resultdir='/export/storage_covidvaccine/Result'):

    path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q/{Chain}'

    evaluate_chain_MNL(Model, Chain, M, K, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, setting_tag, constraint, path)

    if RandomFCFS:
        evaluate_chain_RandomFCFS(Model, Chain, M, K, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, setting_tag, constraint, path)
    if MIP: 
        raise Exception("Warnings: MIP not updated yet \n")
        evaluate_chain_MIP(Model, Chain, M, K, nsplits, capcoef, R, constraint, path)

    return




def evaluate_chain_RandomFCFS(Model,
                              Chain,
                              M,
                              K,
                              nsplits,
                              capcoef,
                              mnl,
                              flexible_consideration,
                              flex_thresh,
                              logdist_above,
                              logdist_above_thresh,
                              R,
                              A,
                              norandomterm,
                              setting_tag,
                              constraint,
                              path):

    print(f'Evaluating random order FCFS with Chain type: {Chain}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}, A = {A}.\n Results stored at {path}\n')
    Chain_dict = {'Dollar': '01_DollarStores', 'Coffee': '04_Coffee', 'HighSchools': '09_HighSchools'}

    z_file_name = f'{path}/{constraint}/z_total'
    z_total = np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)        
    print(f"Import optimization solution from file {z_file_name}{setting_tag}\n")

    if not os.path.exists(f"{path}/{constraint}/ca_blk_{Chain}_dist_total{setting_tag}.csv"):
        print("Distdf not computed for current setting, start computing...\n")
        compute_distdf(Chain_dict[Chain], Chain, constraint, z_total, setting_tag, path)

    if Chain == 'Dollar' and Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': # Pharmacy-only
        block, block_utils, distdf = construct_blocks(Chain, M, K, nsplits, flexible_consideration, flex_thresh, R, A, setting_tag, constraint, path, Pharmacy=True)
        run_assignment(Chain, M, K, nsplits, capcoef, mnl, setting_tag, constraint, block, block_utils, distdf, path, Pharmacy=True)


    block, block_utils, distdf = construct_blocks(Chain, M, K, nsplits, flexible_consideration, flex_thresh, R, A, setting_tag, constraint, path)
    run_assignment(Chain, M, K, nsplits, capcoef, mnl, setting_tag, constraint, block, block_utils, distdf, path)
        
    return




def evaluate_chain_MNL(Model,
                       Chain,
                       M,
                       K,
                       nsplits,
                       capcoef,
                       mnl,
                       flexible_consideration,
                       flex_thresh,
                       logdist_above,
                       logdist_above_thresh,
                       R,
                       A,
                       norandomterm,
                       setting_tag,
                       constraint,
                       path,
                       scale_factor: int = 10000):

    print(f'Evaluating using MNL with setting tag {setting_tag}.\n Results stored at {path}\n')

    (Population, Quartile, abd, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(Chain, M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)
    
    V_current, V_total = import_MNL_estimation(Chain, R, A, setting_tag)
    
    v_total = V_total.flatten()
    pf_total = p_total * v_total
    v_total = v_total * Closest_total
    pf_total = pf_total * Closest_total


    # ================================================================================

    path = f'{path}/{constraint}/'
    if not os.path.exists(path): os.mkdir(path)

    z_file_name = f'{path}z_total'
    z_total = np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)        
    print(f"Import optimization solution from file {z_file_name}{setting_tag}\n")

    evaluate_rate_MNL_partial(C_total=C_total,
                              z=z_total,
                              pf=pf_total,
                              v=v_total,
                              C=C,
                              K=K,
                              closest=Closest_total,
                              num_current_stores=num_current_stores,
                              num_total_stores=num_total_stores,
                              num_tracts=num_tracts,
                              scale_factor=scale_factor,
                              setting_tag=setting_tag,
                              path=path)

    
    if Chain == 'Dollar' and Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': # Pharmacy-only
        
        print("Start evaluating for pharmacies only...\n")
        z_total = np.concatenate((np.ones(num_current_stores), np.zeros(num_total_stores - num_current_stores)))

        evaluate_rate_MNL_partial(C_total=C_total,
                                  z=z_total,
                                  pf=pf_total,
                                  v=v_total,
                                  C=C,
                                  K=K,
                                  closest=Closest_total,
                                  num_current_stores=num_current_stores,
                                  num_total_stores=num_total_stores,
                                  num_tracts=num_tracts,
                                  scale_factor=scale_factor,
                                  setting_tag=setting_tag,
                                  path=path,
                                  Pharmacy=True)


    print("Evaluation finished!\n")

    return




# def evaluate_chain_MIP(Model, Chain, M, K, nsplits, capcoef, R, heuristic, constraint, path, scale_factor=10000):
    
#     print(f'Evaluating MIP with Chain type: {Chain}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}.\n Results stored at {path}\n')
#     Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, _, _, num_tracts, num_current_stores, num_total_stores = import_basics(Chain, M, nsplits)
#     F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain, K, nsplits, capcoef)
    
#     f_dh_current = F_DH_current.flatten()
#     f_dh_total = F_DH_total.flatten()
#     pfdh_current = p_current * f_dh_current
#     pfdh_total = p_total * f_dh_total

#     if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

#         if R is not None: z_total = np.genfromtxt(f'{path}/{constraint}/z_total_fixR{str(R)}.csv', delimiter = ",", dtype = float)
#         else: z_total = np.genfromtxt(f'{path}/{constraint}/z_total.csv', delimiter = ",", dtype = float)

#         evaluate_rate(scenario = 'total', constraint = constraint, z = z_total,
#                     pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
#                     closest = Closest_total, K=K,
#                     num_current_stores=num_current_stores,
#                     num_total_stores=num_total_stores,
#                     num_tracts=num_tracts,
#                     scale_factor=scale_factor,
#                     path = f'{path}/{constraint}/',
#                     R = R)

#     return