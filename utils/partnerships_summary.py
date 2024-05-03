#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu
"""

import os
import pandas as pd
import numpy as np
from tabulate import tabulate

from utils.partnerships_summary_helpers import import_dataset, import_solution, import_solution_leftover, import_locations, import_MNL_basics, create_row_MIP, create_row_MNL_MIP, create_row_randomFCFS, compute_utilization_randomFCFS, export_dist
from utils.import_parameters import import_MNL_estimation



def partnerships_summary(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistLogLin', 'MNL_partial'],
                         Chain_list = ['Dollar', 'HighSchools', 'Coffee'],
                         K: int = 10000, 
                         M: int = 5, 
                         nsplits: int = 4,
                         capcoef: bool = False,
                         flexible_consideration: bool = False,
                         R = None,
                         A = None,
                         random_seed=None,
                         setting_tag: str = '',
                         evaluation: str = 'mnl_mip',
                         constraint='vaccinated', 
                         leftover: bool = True,
                         export_tract: bool = False,
                         export_dist: bool = False,
                         export_utilization: bool = False,
                         suffix: str = '',
                         resultdir: str = '/export/storage_covidvaccine/Result', 
                         datadir: str = '/export/storage_covidvaccine/Data'):

    
    df, df_temp, block, tract_hpi, _ = import_dataset(nsplits, datadir) # non-estimation-related
    chain_summary_table = []

    for Chain in Chain_list:
        print(f'Start computing summary table for Chain = {Chain}...\n')
        
        for Model in Model_list:
            
            if Model == 'MaxVaxDistLogLin' and Chain != 'Dollar': # when we refer to MaxVaxDistLogLin, we really meant Pharmacy-only
                continue

            (pharmacy_locations, chain_locations, num_tracts, num_current_stores, num_total_stores,
            C_current, C_total, C_current_walkable, C_total_walkable) = import_locations(df_temp, Chain)

            print(f'Model = {Model}, K = {K}, M = {M}, with setting tag {setting_tag}...\n')
            path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q/{Chain}/{constraint}/'
            
            # =================================================================================
            
            if evaluation == "random_fcfs":

                # Pharmacy-only
                # if Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': 
                if Chain == 'Dollar' and Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': 
                    z, locs, dists, assignment = import_solution(evaluation, path, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, random_seed, setting_tag, Pharmacy=True)
                    chain_summary = create_row_randomFCFS('Pharmacy-only', Model, Chain, M, K, nsplits, 'none', 'Evaluation', z, block, locs, dists, assignment, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                    chain_summary_table.append(chain_summary)
                else:
                    z, locs, dists, assignment = import_solution(evaluation, path, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, random_seed, setting_tag)   
                    # second stage FCFS
                    chain_summary_second_stage_randomFCFS = create_row_randomFCFS('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'Evaluation', z, block, locs, dists, assignment, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                    chain_summary_table.append(chain_summary_second_stage_randomFCFS)

                # other results
                if export_dist: export_dist(path, Model, Chain, M, K, R, z, block, locs, dists, assignment, chain_locations, num_current_stores, num_total_stores)
                if export_utilization: compute_utilization_randomFCFS(K, R, z, block, locs, dists, assignment, pharmacy_locations, chain_locations, path)
  
            # =================================================================================
                    
            elif evaluation == "mnl_mip":
                
                p_total, C, Closest_total = import_MNL_basics(tract_hpi, C_current, C_total, M, flexible_consideration)
                _, V_total = import_MNL_estimation(Chain, R, A, random_seed, setting_tag)

                v_total = V_total.flatten()
                pf_total = p_total * v_total
                v_total = v_total * Closest_total
                pf_total = pf_total * Closest_total
                
                gamma = (v_total * v_total) / (1 + v_total)
                pg_total = p_total * gamma
                pg_total = pg_total * Closest_total

                # mat_t and mat_f is essentially mat_y and F_DH

                # if Model == 'MNL_partial' and constraint == 'vaccinated': # so that we don't evaluate loglin for sensitivity analysis
                if Model == 'MNL_partial' or Model == 'MNL_partial_new': # so that we don't evaluate loglin for sensitivity analysis
                    
                    if Model == 'MNL_partial_new': z, mat_t, mat_f = import_solution(evaluation, path, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, random_seed, setting_tag, pg_total, v_total, C)
                    else: z, mat_t, mat_f = import_solution(evaluation, path, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, random_seed, setting_tag, pf_total, v_total, C)

                    chain_summary, CA_TRACT = create_row_MNL_MIP('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'Evaluation', 
                                                tract_hpi, mat_t, z, mat_f, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                    chain_summary_table.append(chain_summary)
                    
                    if export_tract:
                        CA_TRACT.to_csv(f'{resultdir}/Sensitivity_results/Tract/CA_TRACT_PharmDollar{setting_tag}{suffix}.csv', encoding='utf-8', index=False, header=True)

                    if leftover:
                        for rank in range(2, 4): # for rank in range(2, M+1):
                            z, mat_t, mat_f = import_solution_leftover(evaluation, path, rank, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, setting_tag)
                            
                            chain_summary, _ = create_row_MNL_MIP('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, f'Evaluation {rank}', 
                                                tract_hpi, mat_t, z, mat_f, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                            chain_summary_table.append(chain_summary)

                # =================================================================================
                        
                if Chain == 'Dollar' and Model == 'MaxVaxDistLogLin' and constraint == 'vaccinated': 
                    
                    MNL_partial_new = True
                    if MNL_partial_new: z, mat_t, mat_f = import_solution(evaluation, path, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, random_seed, setting_tag, pg_total, v_total, C, Pharmacy=True)
                    else: z, mat_t, mat_f = import_solution(evaluation, path, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, random_seed, setting_tag, pf_total, v_total, C, Pharmacy=True)

                    chain_summary, CA_TRACT = create_row_MNL_MIP('Pharmacy-only', Model, Chain, M, K, nsplits, constraint, 'Evaluation', 
                                            tract_hpi, mat_t, z, mat_f, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                    chain_summary_table.append(chain_summary)

                    if export_tract:
                        CA_TRACT.to_csv(f'{resultdir}/Sensitivity_results/Tract/CA_TRACT_Pharm{setting_tag}{suffix}.csv', encoding='utf-8', index=False, header=True)     
                
                    if leftover:
                        for rank in range(2, 4): # for rank in range(2, M+1):
                            z, mat_t, mat_f = import_solution_leftover(evaluation, path, rank, Model, Chain, K, num_tracts, num_total_stores, num_current_stores, setting_tag, Pharmacy=True)
                            
                            chain_summary, _ = create_row_MNL_MIP('Pharmacy-only', Model, Chain, M, K, nsplits, constraint, f'Evaluation {rank}', 
                                                tract_hpi, mat_t, z, mat_f, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                            chain_summary_table.append(chain_summary)


            else:
                # chain_summary_second_stage_MIP = create_row_MIP('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'second stage MIP',
                # tract_hpi, mat_y_eval, z, F_DH_total, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                raise Exception("Evaluation type undefined")

    chain_summary = pd.DataFrame(chain_summary_table)
    if A is not None: chain_summary['A'] = A
    chain_summary.to_csv(f'{resultdir}/Sensitivity_results/Results{setting_tag}{suffix}.csv', encoding='utf-8', index=False, header=True)
    
    print(f"Table exported as Results{setting_tag}{suffix}.csv\n")
    return

