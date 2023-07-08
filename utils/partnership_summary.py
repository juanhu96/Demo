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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
maindir = '/export/storage_covidvaccine/'

from utils.partnerships_summary_helpers import create_row, tract_summary


def partnerships_summary(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist'],
                        Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                        M_list = [5, 10], K_list = [8000, 10000, 12000],
                        Vaccination_estimation = 'BLP', constraint_list = ['assigned', 'vaccinated'], 
                        export_tract_table = False, filename = ''):
    

    '''

    Summary for each (Model, Chain, M, K, opt_constr, eval_constr) pair

    Parameters
    ----------
    Model_list: List of strings
        List of models to construct summary table

    Chain_list: List of strings
        List of partnerships to construct summary table

    K_list: List of int
        List of capacity to construct summary table
        For 'MinDist' this is only feasible for K = 10000, 12000

    Vaccination_estimation : string
        Demand function used for computing predicted vaccination
        'BLP', 'linear'

    export_tract_table: Boolean
        Whether to export a tract-level summary table (for diagonsis)

    filename : string
        Filename
    
    '''

    
    chain_summary_table = []


    ## Demand coefficients
    # Demand_parameter = [[1.227, -0.452], [2.028, -0.120, -1.357, -1.197, -0.925, -0.254, -0.218, -0.114]] # v3
    if Vaccination_estimation == 'BLP': Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]] # v2
    if Vaccination_estimation == 'linear': Demand_parameter = [[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, 0.053, 0.047, 0.039]]
    

    ## Population and quartiles for CA
    Quartile = np.genfromtxt(f'{maindir}/Data/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)    
    Population = np.genfromtxt(f'{maindir}/Data/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    

    ## Distance matrix for pharmacies (current)
    C_current = np.genfromtxt(f'{maindir}/Data/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current = C_current.astype(int)
    C_current = C_current.T
    num_tracts, num_current_stores = np.shape(C_current)
   

    for Model in Model_list:
        for Chain_type in Chain_list:


            ## Distance matrix for chain
            C_chains = np.genfromtxt(f'{maindir}/Data/CA_dist_matrix_{Chain_type}.csv', delimiter = ",", dtype = float)
            C_chains = C_chains.astype(int)
            C_chains = C_chains.T
            num_tracts, num_chain_stores = np.shape(C_chains)
            C_chains = np.where(C_chains < 0, 1317574, C_chains) # High schools
            
            C_total = np.concatenate((C_current, C_chains), axis = 1)
            num_total_stores = num_current_stores + num_chain_stores
            
            C_current_walkable = np.where(C_current < 1600, 1, 0)
            C_chains_walkable = np.where(C_chains < 1600, 1, 0)
            C_total_walkable = np.where(C_total < 1600, 1, 0)
            

            ## Demand matrix for chain
            if Vaccination_estimation == 'BLP': F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_BLP(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)
            if Vaccination_estimation == 'linear': F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)

            
            for M in M_list:

                # M closest stores only
                Closest_current = np.ones((num_tracts, num_current_stores))
                Closest_total = np.ones((num_tracts, num_total_stores))
                np.put_along_axis(Closest_current, np.argpartition(C_current,M,axis=1)[:,M:],0,axis=1)
                np.put_along_axis(Closest_total, np.argpartition(C_total,M,axis=1)[:,M:],0,axis=1)
                                            
                Closest_total_chains = np.ones((num_tracts, num_chain_stores))
                np.put_along_axis(Closest_total_chains, np.argpartition(C_chains,M,axis=1)[:,M:],0,axis=1)
                Closest_total_chains = np.concatenate((Closest_current, Closest_total_chains), axis = 1)

                Farthest_current = np.ones((num_tracts, num_current_stores)) - Closest_current
                Farthest_total_chains = np.ones((num_tracts, num_total_stores)) - Closest_total

                Closest_current = Closest_current.flatten()
                Closest_total_chains = Closest_total.flatten()

                Farthest_current = Farthest_current.flatten()
                Farthest_total_chains = Farthest_total_chains.flatten()


                ###########################################################################


                for K in K_list:  

                    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

                        for opt_constr in constraint_list:

                            path = f'{maindir}/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/{opt_constr}/'

                            for eval_constr in constraint_list:

                                y_hpi = np.genfromtxt(f'{path}y_total_eval_{eval_constr}.csv', delimiter = ",", dtype = float)
                                z_hpi = np.genfromtxt(f'{expdirpath}{opt_constr}/z_total.csv', delimiter = ",", dtype = float) # invariant of eval_constr

                                y_hpi_closest = y_hpi * Closest_chains
                                y_hpi_farthest = y_hpi * Farthest_chains

                                mat_y_hpi = np.reshape(y_hpi, (num_tracts, num_stores))
                                mat_y_hpi_closest = np.reshape(y_hpi_closest, (num_tracts, num_stores))
                                mat_y_hpi_farthest = np.reshape(y_hpi_farthest, (num_tracts, num_total_stores))

                                R = store_used - sum(z_hpi[0 : num_current_stores])

                                chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, eval_constr,\
                                Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_total, C_total, C_total_walkable)

                                chain_summary_table.append(chain_summary)


                                if Chain_type == 'Dollar':

                                    y_hpi = np.genfromtxt(f'{path}y_current_eval_{eval_constr}.csv', delimiter = ",", dtype = float)

                                    y_hpi_closest = y_hpi * Closest_chains
                                    y_hpi_farthest = y_hpi * Farthest_chains

                                    mat_y_hpi = np.reshape(y_hpi, (num_tracts, num_stores))
                                    mat_y_hpi_closest = np.reshape(y_hpi_closest, (num_tracts, num_stores))
                                    mat_y_hpi_farthest = np.reshape(y_hpi_farthest, (num_tracts, num_total_stores))

                                    R = 0

                                    chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, opt_constr, eval_constr,\
                                    Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_current, C_current, C_current_walkable)

                                    chain_summary_table.append(chain_summary)

                                    # if export_tract_table==True:

                                    #     tract_summary(Model, Chain_type, M, K, Quartile, Population,\
                                    #     mat_y_current_hpi, mat_y_current_hpi_closest, mat_y_current_hpi_farthest, result_current_hpi, F_DH_current, C_current, C_current_walkable,\
                                    #     mat_y_total_hpi, mat_y_total_hpi_closest, mat_y_total_hpi_farthest, result_total_hpi, F_DH_total, C_total, C_total_walkable)

                    

                    else: # MinDist, no opt_constr

                            path = f'{maindir}/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'

                            for eval_constr in constraint_list:

                                y_hpi = np.genfromtxt(f'{path}y_total_eval_{eval_constr}.csv', delimiter = ",", dtype = float)
                                z_hpi = np.genfromtxt(f'{expdirpath}{opt_constr}/z_total.csv', delimiter = ",", dtype = float)

                                y_hpi_closest = y_hpi * Closest_chains
                                y_hpi_farthest = y_hpi * Farthest_chains

                                mat_y_hpi = np.reshape(y_hpi, (num_tracts, num_stores))
                                mat_y_hpi_closest = np.reshape(y_hpi_closest, (num_tracts, num_stores))
                                mat_y_hpi_farthest = np.reshape(y_hpi_farthest, (num_tracts, num_total_stores))

                                R = store_used - sum(z_hpi[0 : num_current_stores])

                                chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, eval_constr,\
                                Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_total, C_total, C_total_walkable)

                                chain_summary_table.append(chain_summary)


                                # Current
                                if Chain_type == 'Dollar':

                                    y_hpi = np.genfromtxt(f'{path}y_current_eval_{eval_constr}.csv', delimiter = ",", dtype = float)

                                    y_hpi_closest = y_hpi * Closest_chains
                                    y_hpi_farthest = y_hpi * Farthest_chains

                                    mat_y_hpi = np.reshape(y_hpi, (num_tracts, num_stores))
                                    mat_y_hpi_closest = np.reshape(y_hpi_closest, (num_tracts, num_stores))
                                    mat_y_hpi_farthest = np.reshape(y_hpi_farthest, (num_tracts, num_total_stores))

                                    R = 0

                                    chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, opt_constr, eval_constr,\
                                    Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_current, C_current, C_current_walkable)

                                    chain_summary_table.append(chain_summary)

                    
                    
                        
    chain_summary = pd.DataFrame(chain_summary_table)
    chain_summary.to_csv('../Result/summary_table_demand_.csv', encoding='utf-8', index=False, header=True)
                
                








if __name__ == "__main__":
    main()    
