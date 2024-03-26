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

from utils.evaluate_model import compute_distdf, construct_blocks, run_assignment, evaluate_rate, evaluate_rate_MNL_partial, evaluate_rate_MNL_partial_leftover, compute_f, update_f, compute_g, update_g
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
                  random_seed,
                  setting_tag: str,
                  RandomFCFS: bool = False,
                  leftover: bool = True,
                  constraint='vaccinated',
                  resultdir='/export/storage_covidvaccine/Result'):

    def create_path(Model, Chain, K, M, nsplits, resultdir):
        
        model_path = f'{resultdir}/{Model}'
        if not os.path.exists(model_path): os.mkdir(model_path)
        parameter_path = f'{model_path}/M{str(M)}_K{str(K)}_{nsplits}q'
        if not os.path.exists(parameter_path): os.mkdir(parameter_path)
        chain_path = f'{parameter_path}/{Chain}/'
        if not os.path.exists(chain_path): os.mkdir(chain_path)
        return chain_path
       
    path = create_path(Model, Chain, K, M, nsplits, resultdir)

    evaluate_chain_MNL(Model, Chain, M, K, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, random_seed, setting_tag, constraint, path)
    
    if leftover: 
        print("Evaluation with leftover fulfillments...\n")
        for rank in range(2, 4): # computation/storage issue, for rank in range(2, M+1)
            evalute_chain_MNL_leftover(Model, Chain, M, K, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, random_seed, setting_tag, constraint, path, rank)
    
    
    # other evaluation form
    if RandomFCFS:
        print("Evaluation with random FCFS...\n")
        evaluate_chain_RandomFCFS(Model, Chain, M, K, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, setting_tag, constraint, path)

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
                       random_seed,
                       setting_tag,
                       constraint,
                       path,
                       scale_factor: int = 10000):

    print(f'Evaluating using MNL with setting tag {setting_tag}.\n Results stored at {path}\n')

    (Population, Quartile, abd, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(Chain, M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)
    
    _, V_total = import_MNL_estimation(Chain, R, A, random_seed, setting_tag)
    v_total = V_total.flatten()
    v_total = v_total * Closest_total
    pv_total = p_total * v_total

    # ================================================================================

    path = f'{path}/{constraint}/'
    if not os.path.exists(path): os.mkdir(path)
    
    if Model == 'MNL_partial' or Model == 'MNL_partial_new':

        z_file_name = f'{path}z_total'
        z_total = np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float) 
        print(f'The number of pharmacies opened: {np.sum(z_total[:num_current_stores] == 1)}, number of {Chain} opened: {np.sum(z_total[num_current_stores:] == 1)}')
        
        non_binary_check = np.any((z_total != 0) & (z_total != 1))
        print(f'Is there any fractional number in the final solution z: {non_binary_check}')
        if non_binary_check:
            print('The fractional numbers are')
            non_binary_indices = np.where((z_total != 0) & (z_total != 1))
            non_binary_values = z_total[non_binary_indices]
            for index, value in zip(non_binary_indices[0], non_binary_values):
                print(f"index: {index}, value: {value}")

            z_total = z_total.astype(int)

        if Model == 'MNL_partial_new':
            print("================== MNL_partial_new ==================\n")
            gamma = (v_total * v_total) / (1 + v_total)
            pg_total = p_total * gamma
            pg_total = pg_total * Closest_total
            f = compute_g(z_total, pg_total, v_total, C, num_total_stores, num_tracts)
        else: 
            f = compute_f(z_total, pv_total, v_total, C, num_total_stores, num_tracts)

        print(f'B/C consideration set, the maximum possible demand in coming round is {np.round(np.sum(f) / 1000000, 5)};\n')

        # NOTE: for sensitivity analysis we don't need to evalaute for MaxVaxDistLogLin
        evaluate_rate_MNL_partial(f=f,
                                z=z_total,
                                K=K,
                                closest=Closest_total,
                                num_current_stores=num_current_stores,
                                num_total_stores=num_total_stores,
                                num_tracts=num_tracts,
                                setting_tag=setting_tag,
                                path=path)
        


    if Chain == 'Dollar' and Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': # Pharmacy-only
        
        print("Start evaluating for pharmacies only...\n")
        if A is not None:
            np.random.seed(random_seed)
            print(f"WE ARE HERE WITH A = {A} and RANDOM SEED OF {random_seed}\n")
            selected_dollar = np.zeros(num_total_stores - num_current_stores)
            indices = np.random.choice(range(len(selected_dollar)), A, replace=False)
            print(indices)
            selected_dollar[indices] = 1
            z_total = np.concatenate((np.ones(num_current_stores), selected_dollar))
            np.savetxt(f'{path}z_Pharmacy_round{1}{setting_tag}.csv', z_total, delimiter=",")
        else:
            z_total = np.concatenate((np.ones(num_current_stores), np.zeros(num_total_stores - num_current_stores)))

        print(f'The number of pharmacies opened: {np.sum(z_total[:num_current_stores] == 1)}, number of {Chain} opened: {np.sum(z_total[num_current_stores:] == 1)}')

        # f = compute_f(z_total, pv_total, v_total, C, num_total_stores, num_tracts)
        
        gamma = (v_total * v_total) / (1 + v_total)
        pg_total = p_total * gamma
        pg_total = pg_total * Closest_total
        f = compute_g(z_total, pg_total, v_total, C, num_total_stores, num_tracts)

        print(f'B/C consideration set, the maximum possible demand in coming round is {np.round(np.sum(f) / 1000000, 5)};\n')

        evaluate_rate_MNL_partial(f=f,
                                  z=z_total,
                                  K=K,
                                  closest=Closest_total,
                                  num_current_stores=num_current_stores,
                                  num_total_stores=num_total_stores,
                                  num_tracts=num_tracts,
                                  setting_tag=setting_tag,
                                  path=path,
                                  Pharmacy=True)

    print("Evaluation finished!\n")

    return




def evalute_chain_MNL_leftover(Model,
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
                               random_seed,
                               setting_tag,
                               constraint,
                               path,
                               rank,
                               scale_factor: int = 10000):
    print("="*120)
    print(f'Start filling the leftover demands to their {rank} choice\n')

    (Population, Quartile, abd, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(Chain, M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)
    
    _, V_total = import_MNL_estimation(Chain, R, A, random_seed, setting_tag)
    v_total = V_total.flatten()
    v_total = v_total * Closest_total
    
    # ================================================================================

    path = f'{path}/{constraint}/'
    if not os.path.exists(path): os.mkdir(path)
    
    if Model == 'MNL_partial' or Model == 'MNL_partial_new': # Pharmacy + Chain  

        z_file_name = f'{path}z_total' if rank == 2 else f'{path}z_total_round{rank-1}'
        Kz_file_name = f'{path}Kz_round{rank-1}'
        t_file_name = f'{path}t' if rank == 2 else f'{path}t_round{rank-1}'
        f_file_name = f'{path}f_round{rank-1}'

        # z_prev: locations with leftover capacity from previous round
        # Kz_prev: capacity from previous round
        # t_prev: fulfillment from previous round (to compute unmet demand)
        # p_prev: leftover population from previous round
        z_prev = np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)    
        Kz_prev = K * z_prev if rank == 2 else np.genfromtxt(f'{Kz_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
        t_prev = np.genfromtxt(f'{t_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
        
        if Model == 'MNL_partial_new':
            print("================== MNL_partial_new ==================\n")
            gamma = (v_total * v_total) / (1 + v_total)
            pg_total = p_total * gamma
            pg_total = pg_total * Closest_total
            f_prev = compute_g(z_prev, pg_total, v_total, C, num_total_stores, num_tracts) if rank == 2 else np.genfromtxt(f'{f_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
            Kz_new, z_new, f_new, terminate = update_g(f_prev, z_prev, t_prev, v_total, gamma, C, Kz_prev, num_total_stores, num_tracts)
        else:
            f_prev = compute_f(z_prev, p_total * v_total, v_total, C, num_total_stores, num_tracts) if rank == 2 else np.genfromtxt(f'{f_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
            Kz_new, z_new, f_new, terminate = update_f(f_prev, z_prev, t_prev, v_total, C, Kz_prev, num_total_stores, num_tracts)

        if terminate: return

        # takes in updated f/Kz/z, saves z_new, f_new and compute t_new
        evaluate_rate_MNL_partial_leftover(f=f_new,
                                        Kz=Kz_new,
                                        z=z_new,
                                        closest=Closest_total,
                                        num_current_stores=num_current_stores,
                                        num_total_stores=num_total_stores,
                                        num_tracts=num_tracts,
                                        scale_factor=scale_factor,
                                        setting_tag=setting_tag,
                                        path=path,
                                        rank=rank)


    if Chain == 'Dollar' and Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': # Pharmacy-only
        
        z_file_name = f'{path}z_Pharmacy_round{rank-1}'
        Kz_file_name = f'{path}Kz_Pharmacy_round{rank-1}'
        t_file_name = f'{path}t_Pharmacy' if rank == 2 else f'{path}t_Pharmacy_round{rank-1}'
        f_file_name = f'{path}f_Pharmacy_round{rank-1}'
        
        if A is not None:
            z_prev = np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float) # b/c now round 1 is from randomization  
        else:    
            z_prev = np.concatenate((np.ones(num_current_stores), np.zeros(num_total_stores - num_current_stores))) if rank == 2 else np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)    
        Kz_prev = K * z_prev if rank == 2 else np.genfromtxt(f'{Kz_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
        t_prev = np.genfromtxt(f'{t_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
        # f_prev = compute_f(z_prev, p_total * v_total, v_total, C, num_total_stores, num_tracts) if rank == 2 else np.genfromtxt(f'{f_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
        # Kz_new, z_new, f_new, terminate  = update_f(f_prev, z_prev, t_prev, v_total, C, Kz_prev, num_total_stores, num_tracts)

        gamma = (v_total * v_total) / (1 + v_total)
        pg_total = p_total * gamma
        pg_total = pg_total * Closest_total
        f_prev = compute_g(z_prev, pg_total, v_total, C, num_total_stores, num_tracts) if rank == 2 else np.genfromtxt(f'{f_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)
        Kz_new, z_new, f_new, terminate = update_g(f_prev, z_prev, t_prev, v_total, gamma, C, Kz_prev, num_total_stores, num_tracts)

        if terminate: return

        evaluate_rate_MNL_partial_leftover(f=f_new,
                                           Kz=Kz_new,
                                           z=z_new,
                                           closest=Closest_total,
                                           num_current_stores=num_current_stores,
                                           num_total_stores=num_total_stores,
                                           num_tracts=num_tracts,
                                           scale_factor=scale_factor,
                                           setting_tag=setting_tag,
                                           path=path,
                                           rank=rank,
                                           Pharmacy=True)


    return
