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

from utils.evaluate_model import compute_distdf, construct_blocks, run_assignment, evaluate_rate



def evaluate_main(Model, Chain, M, K, groups, capcoef, R=None, MIP=False, constraint='vaccinated', resultdir='/export/storage_covidvaccine/Result'):

    if capcoef: path = f'{resultdir}/M{str(M)}_K{str(K)}_{groups}q_capcoef/{Chain}'
    else: path = f'{resultdir}/M{str(M)}_K{str(K)}_{groups}q/{Chain}'

    # path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}/{Chain}'
    
    evaluate_chain_RandomFCFS(Model, Chain, M, K, R, constraint, path)
    if MIP: evaluate_chain_MIP(Model, Chain, M, K, R, constraint, path)

    return




def evaluate_chain_RandomFCFS(Model, Chain, M, K, R, constraint, path):

    print(f'Evaluating random order FCFS with Chain type: {Chain}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}.\n Results stored at {path}\n')
    Chain_dict = {'Dollar': '01_DollarStores', 'Coffee': '04_Coffee', 'HighSchools': '09_HighSchools'}
    
    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:
            
        if R is not None: z_total = np.genfromtxt(f'{path}/{constraint}/z_total_fixR{str(R)}.csv', delimiter = ",", dtype = float)
        else: z_total = np.genfromtxt(f'{path}/{constraint}/z_total.csv', delimiter = ",", dtype = float)
            
        compute_distdf(Chain_dict[Chain], Chain, constraint, z_total, R, path)

        if Chain == 'Dollar' and Model == 'MaxVaxHPIDistBLP' and constraint == 'vaccinated': # only once
            # Pharmacy-only
            block, block_utils, distdf = construct_blocks(Chain, M, K, R, constraint, path, Pharmacy=True)
            run_assignment(Chain, M, K, R, constraint, block, block_utils, distdf, path, Pharmacy=True)
                
        # Pharmacy+others
        block, block_utils, distdf = construct_blocks(Chain, M, K, R, constraint, path)
        run_assignment(Chain, M, K, R, constraint, block, block_utils, distdf, path)
            
    else: # MinDist

        z_total = np.genfromtxt(f'{path}z_total.csv', delimiter = ",", dtype = float)
        compute_distdf(chain_type=Chain_dict[Chain], chain_name=Chain, constraint='None', z=z_total, expdirpath=path)

        block, block_utils, distdf = construct_blocks(Chain, M, K, 'None', path)
        run_assignment(Chain, M, K, 'None', block, block_utils, distdf, path)


    return




def evaluate_chain_MIP(Model, Chain, M, K, R, constraint, path):
    
    print(f'Evaluating MIP with Chain type: {Chain_type}; Model: {Model}; M = {str(M)}, K = {str(K)}, R = {R}. Results stored at {path}\n')
    Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, _, _, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type, M)
    F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain_type, K)
    
    f_dh_current = F_DH_current.flatten()
    f_dh_total = F_DH_total.flatten()
    pfdh_current = p_current * f_dh_current
    pfdh_total = p_total * f_dh_total

    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

        z_total = np.genfromtxt(f'{expdirpath}{constraint}/z_total.csv', delimiter = ",", dtype = float)

        if Chain_type == 'Dollar':
                
            z_current = np.genfromtxt(f'{expdirpath}{constraint}/z_current.csv', delimiter = ",", dtype = float) # only solved once during Dollar
                
            evaluate_rate(scenario = 'current', constraint = constraint, z = z_current,
                        pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                        closest = Closest_current, K=K, 
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores, 
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath + constraint + '/')

        evaluate_rate(scenario = 'total', constraint = constraint, z = z_total,
                    pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                    closest = Closest_total, K=K,
                    num_current_stores=num_current_stores,
                    num_total_stores=num_total_stores,
                    num_tracts=num_tracts,
                    scale_factor=scale_factor,
                    path = expdirpath + constraint + '/')


    else: # MinDist

        z_total = np.genfromtxt(f'{expdirpath}z_total.csv', delimiter = ",", dtype = float)
        z_current = np.genfromtxt(f'{expdirpath}z_current.csv', delimiter = ",", dtype = float)
        
        if Chain_type == 'Dollar':
            evaluate_rate(scenario = 'current', constraint = constraint, z = z_current,
                        pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                        closest = Closest_current, K=K, 
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores, 
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath, 
                        MIPGap = 1e-2)

        evaluate_rate(scenario = 'total', constraint = constraint, z = z_total,
                    pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                    closest = Closest_total, K=K,
                    num_current_stores=num_current_stores,
                    num_total_stores=num_total_stores,
                    num_tracts=num_tracts,
                    scale_factor=scale_factor,
                    path = expdirpath,
                    MIPGap = 5e-2)


    return