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

from utils.evaluate_model import compute_distdf, construct_blocks, run_assignment, evaluate_rate
from utils.import_parameters import import_basics, import_BLP_estimation


def evaluate_main(Model, Chain, M, K, nsplits, capcoef, R=None, heuristic=False, MIP=False, constraint='vaccinated', resultdir='/export/storage_covidvaccine/Result'):


    if capcoef: path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef/{Chain}'
    else: path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q/{Chain}'
    

    evaluate_chain_RandomFCFS(Model, Chain, M, K, nsplits, R, heuristic, constraint, path)
    if MIP: evaluate_chain_MIP(Model, Chain, M, K, nsplits, capcoef, R, heuristic, constraint, path)

    return




def evaluate_chain_RandomFCFS(Model, Chain, M, K, nsplits, R, heuristic, constraint, path):

    print(f'Evaluating random order FCFS with Chain type: {Chain}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}, Heuristic: {heuristic}.\n Results stored at {path}\n')
    Chain_dict = {'Dollar': '01_DollarStores', 'Coffee': '04_Coffee', 'HighSchools': '09_HighSchools'}

    z_file_name = f'{path}/{constraint}/z_total'
    if R is not None: z_file_name += f'_fixR{str(R)}'
    if heuristic: z_file_name += '_heuristic'
    z_total = np.genfromtxt(f'{z_file_name}.csv', delimiter = ",", dtype = float)        
        
    compute_distdf(Chain_dict[Chain], Chain, constraint, z_total, R, heuristic, path) # NOTE: JUST FOR NOW

    if Chain == 'Dollar' and Model == 'MaxVaxHPIDistBLP' and constraint == 'vaccinated': # Pharmacy-only
        block, block_utils, distdf = construct_blocks(Chain, M, K, nsplits, R, heuristic, constraint, path, Pharmacy=True)
        run_assignment(Chain, M, K, R, heuristic, constraint, block, block_utils, distdf, path, Pharmacy=True)


    block, block_utils, distdf = construct_blocks(Chain, M, K, nsplits, R, heuristic, constraint, path)
    run_assignment(Chain, M, K, R, heuristic, constraint, block, block_utils, distdf, path)
            

    return




def evaluate_chain_MIP(Model, Chain, M, K, nsplits, capcoef, R, heuristic, constraint, path, scale_factor=10000):
    
    print(f'Evaluating MIP with Chain type: {Chain}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}.\n Results stored at {path}\n')
    Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, _, _, num_tracts, num_current_stores, num_total_stores = import_basics(Chain, M, nsplits)
    F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain, K, nsplits, capcoef)
    
    f_dh_current = F_DH_current.flatten()
    f_dh_total = F_DH_total.flatten()
    pfdh_current = p_current * f_dh_current
    pfdh_total = p_total * f_dh_total

    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

        if R is not None: z_total = np.genfromtxt(f'{path}/{constraint}/z_total_fixR{str(R)}.csv', delimiter = ",", dtype = float)
        else: z_total = np.genfromtxt(f'{path}/{constraint}/z_total.csv', delimiter = ",", dtype = float)

        evaluate_rate(scenario = 'total', constraint = constraint, z = z_total,
                    pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                    closest = Closest_total, K=K,
                    num_current_stores=num_current_stores,
                    num_total_stores=num_total_stores,
                    num_tracts=num_tracts,
                    scale_factor=scale_factor,
                    path = f'{path}/{constraint}/',
                    R = R)


    return