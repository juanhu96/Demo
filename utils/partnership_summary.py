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

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

from utils.partnerships_summary_helpers import import_dataset, import_solution, import_locations, create_row_first_stage, create_row_second_stage
from utils.import_demand import import_BLP_estimation


def partnerships_summary(Model_list=['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist'],
                        Chain_list=['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                        M_list=[5, 10], K_list=[8000, 10000, 12000], constraint_list=['assigned', 'vaccinated'], nsplits=3,
                        resultdir='/export/storage_covidvaccine/Result/', datadir='/export/storage_covidvaccine/Data/', filename=''):

    '''
    Summary for each (Model, Chain, M, K, opt_constr) pair

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

    df, df_temp, block, tract_hpi = import_dataset(nsplits, datadir)
    print(sum(block.population), sum(tract_hpi.population)) # 39million vs. 34million
    chain_summary_table = []

    for Model in Model_list:
        for Chain_type in Chain_list:

            pharmacy_locations, chain_locations, num_tracts, num_current_stores, num_total_stores, C_current, C_total, C_current_walkable, C_total_walkable = import_locations(df_temp, Chain_type)

            for M in M_list:
                for K in K_list:
                    
                    F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain_type, K)

                    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

                        for opt_constr in constraint_list:

                            path = f'{resultdir}{Model}/M{str(M)}_K{str(K)}/{Chain_type}/{opt_constr}/'
                            z, mat_y, dists, assignment = import_solution(path, Chain_type, K, num_tracts, num_total_stores)
                            
                            chain_summary_first_stage = create_row_first_stage('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, 'first stage', tract_hpi, mat_y, z, F_DH_total, C_total, C_total_walkable, pharmacy_locations, chain_locations, num_current_stores)
                            chain_summary_table.append(chain_summary_first_stage)

                            chain_summary_second_stage = create_row_second_stage('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, 'second stage', z, block, dists, assignment, pharmacy_locations, chain_locations, num_current_stores)
                            chain_summary_table.append(chain_summary_second_stage)

                            if Chain_type == 'Dollar' and Model == 'MaxVaxHPIDistBLP' and opt_constr == 'assigned': # Pharmacy-only
                                    
                                path = f'{resultdir}{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'
                                dists, assignment = import_solution(path, Chain_type, K, num_tracts, num_total_stores, True)

                                ### TODO: compute the y* under pharmacy-only for first-stage results
                                # create_row_first_stage()

                                chain_summary = create_row_second_stage('Pharmacy-only', Model, Chain_type, M, K, 'none', 'second stage', z, block, dists, assignment, pharmacy_locations, chain_locations, num_current_stores)
                                chain_summary_table.append(chain_summary)

                    else: # MinDist

                        path = f'{resultdir}{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'

                        z, y, dists, assignment = import_solution(path, Chain_type, K)

                        chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, z, block, dists, assignment, pharmacy_locations, chain_locations)
                        chain_summary_table.append(chain_summary)

                        if Chain_type == 'Dollar':

                            dists, assignment = import_solution(path, Chain_type, K, True)

                            chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, "none", z, block, dists, assignment, pharmacy_locations, chain_locations)
                            chain_summary_table.append(chain_summary)


    chain_summary = pd.DataFrame(chain_summary_table)
    chain_summary.to_csv(f'{resultdir}Tables/sensitivity_results_{filename}.csv', encoding='utf-8', index=False, header=True)

