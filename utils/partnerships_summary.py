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

from utils.partnerships_summary_helpers import import_dataset, import_solution, import_locations, create_row_MIP, create_row_randomFCFS, compute_utilization_randomFCFS, export_dist
from utils.import_parameters import import_BLP_estimation



def partnerships_summary(Model_list=['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist'],
                        Chain_list=['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                        M_list=[5, 10], 
                        K_list=[8000, 10000, 12000], 
                        nsplits=4,
                        capcoef=True,
                        R=None,
                        heuristic=False,
                        constraint='vaccinated', 
                        second_stage_MIP=False,
                        export_dist=False,
                        export_utilization=False,
                        resultdir='/export/storage_covidvaccine/Result', 
                        datadir='/export/storage_covidvaccine/Data', 
                        filename=''):

    '''
    Summary for each (Model, Chain, M, K, constraint) pair

    Parameters
    ----------
    Model_list: List of strings
        List of models to construct summary table

    Chain_list: List of strings
        List of partnerships to construct summary table

    K_list: List of int
        List of capacity to construct summary table
        For 'MinDist' this is only feasible for K = 10000, 12000

    constraint_list: List of strings
        List of constraint types that are used in first-stage optimization

    nsplits: Int
        Number of splits for HPI quantiles, default = 3

    filename : string
        Filename
    '''
    
    
    df, df_temp, block, tract_hpi, _ = import_dataset(nsplits, datadir)
    chain_summary_table = []

    for Model in Model_list:
        for Chain in Chain_list:

            pharmacy_locations, chain_locations, num_tracts, num_current_stores, num_total_stores, C_current, C_total, C_current_walkable, C_total_walkable = import_locations(df_temp, Chain)

            for M in M_list:
                for K in K_list:
                    
                    print(f'Model={Model}, M={M}, K={K}, Chain={Chain}, nsplits={nsplits}, capcoef={capcoef}, R={R}, heuristic={heuristic}\n')
                    
                    F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain, K, nsplits, capcoef)

                    if capcoef: path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef/{Chain}/{constraint}/'
                    else: path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q/{Chain}/{constraint}/'

                    z, mat_y, mat_y_eval, locs, dists, assignment = import_solution(path, Chain, K, num_tracts, num_total_stores, num_current_stores, R)
                

                    # first stage MIP
                    chain_summary_first_stage = create_row_MIP('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'first stage', tract_hpi, mat_y, z, F_DH_total, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                    chain_summary_table.append(chain_summary_first_stage)


                    # second stage MIP
                    if second_stage_MIP:
                        chain_summary_second_stage_MIP = create_row_MIP('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'second stage MIP', tract_hpi, mat_y_eval, z, F_DH_total, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                        chain_summary_table.append(chain_summary_second_stage_MIP)


                    # second stage FCFS
                    chain_summary_second_stage_randomFCFS = create_row_randomFCFS('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'second stage randomFCFS', z, block, locs, dists, assignment, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                    chain_summary_table.append(chain_summary_second_stage_randomFCFS)


                    # other results
                    if export_dist: export_dist(path, Model, Chain, M, K, R, z, block, locs, dists, assignment, chain_locations, num_current_stores, num_total_stores)
                    if export_utilization: compute_utilization_randomFCFS(K, R, z, block, locs, dists, assignment, pharmacy_locations, chain_locations, path)


                    # heuristic results
                    if Model == "MaxVaxHPIDistBLP" and heuristic:
                                
                        z, mat_y, mat_y_eval, locs, dists, assignment = import_solution(path, Chain, K, num_tracts, num_total_stores, num_current_stores, R, heuristic)

                        # second stage MIP
                        if second_stage_MIP:
                            chain_summary_second_stage_MIP = create_row_MIP('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'second stage MIP (heuristic)', tract_hpi, mat_y_eval, z, F_DH_total, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                            chain_summary_table.append(chain_summary_second_stage_MIP)

                        # second stage FCFS
                        chain_summary_second_stage_randomFCFS = create_row_randomFCFS('Pharmacy + ' + Chain, Model, Chain, M, K, nsplits, constraint, 'second stage randomFCFS (heuristic)', z, block, locs, dists, assignment, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                        chain_summary_table.append(chain_summary_second_stage_randomFCFS)


                    if Chain == 'Dollar' and Model == 'MaxVaxHPIDistBLP' and constraint == 'vaccinated': # Pharmacy-only
                                
                        # z, mat_y, mat_y_eval, locs, dists, assignment = import_solution(path, Chain, K, num_tracts, num_total_stores, num_current_stores, R, heuristic=False, Pharmacy=True)
                        z, locs, dists, assignment = import_solution(path, Chain, K, num_tracts, num_total_stores, num_current_stores, R, heuristic=False, Pharmacy=True)

                        # chain_summary_first_stage = create_row_MIP('Pharmacy-only', Model, Chain, M, K, nsplits, 'none', 'first stage', tract_hpi, mat_y, z, F_DH_current, C_current, C_current_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                        # chain_summary_table.append(chain_summary_first_stage)

                        chain_summary = create_row_randomFCFS('Pharmacy-only', Model, Chain, M, K, nsplits, 'none', 'second stage randomFCFS', z, block, locs, dists, assignment, pharmacy_locations, chain_locations, num_current_stores, num_total_stores)
                        chain_summary_table.append(chain_summary)
                    


    chain_summary = pd.DataFrame(chain_summary_table)
    if R is not None: chain_summary.to_csv(f'{resultdir}/Sensitivity_results/sensitivity_results_{filename}R{R}.csv', encoding='utf-8', index=False, header=True)
    else: chain_summary.to_csv(f'{resultdir}/Sensitivity_results/sensitivity_results_{filename}.csv', encoding='utf-8', index=False, header=True)

