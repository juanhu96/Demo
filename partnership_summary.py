#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 19, 2023
@author: Jingyuan Hu
"""

import os
import pandas as pd
import numpy as np
from tabulate import tabulate
os.chdir('/export/storage_covidvaccine/Code')
from utils.create_row import create_row, tract_summary


def main():

    print('Start creating summary table...')

    create_summary(Model_list = ['MaxRateHPIDist', 'MaxRateDist', 'MaxRateFixV'], Chain_list = ['Dollar'], Demand_estimation = 'linear') # optimize w/ linear, compute w/ blp
    create_summary(Model_list = ['MaxRateHPIDist', 'MaxRateDist', 'MaxRateFixV'], Chain_list = ['Dollar'], Demand_estimation = 'BLP') # optimize w/ blp, compute w/ blp
    create_summary(Model_list = ['MinDistNew'], Chain_list = ['Dollar'],  K_list = [10000, 12000], filename = 'MinDist')


    print('Finish creating the table!')


###########################################################################
###########################################################################
###########################################################################


def create_summary(M_list = [5, 10], K_list = [8000, 10000, 12000],
                   Model_list = ['MaxRateHPIDist', 'MaxRateDist', 'MaxRateFixV', 'MinDist', 'MinDistNew'],
                   Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                   Demand_estimation = 'BLP', export_tract_table = False, filename = ''):
    

    '''
    Parameters
    ----------
    Model_list: List of strings
        List of models to construct summary table

    Chain_list: List of strings
        List of partnerships to construct summary table

    K_list: List of int
        List of capacity to construct summary table
        For 'MinDist' this is only feasible for K = 10000, 12000

    Demand_estimation : string
        Demand function used when solving the optimization problem, applicable to 'MaxRateHPIDist' and 'MaxRateDist' only
        "BLP", "Linear", "Logit" 

        For computing vaccination we will always apply the final BLP model

    export_tract_table: Boolean
        Whether to export a tract-level summary table (for diagonsis)

    filename : string
        Filename
    
    '''

    
    chain_summary_table = []
    

    ## Demand coefficients
    datadir = "/export/storage_covidvaccine/Data"
    m1coefs = np.load(f'{datadir}/Analysis/m1coefs.npy')
    m2coefs = np.load(f'{datadir}/Analysis/m2coefs.npy')
    # Demand_parameter = [np.round(m1coefs, 3).tolist(), np.round(m2coefs, 3).tolist()]
    # Demand_parameter = [[1.144, -0.567], [1.676, -0.243, -1.101, -0.796, -0.699, -0.331, -0.343, -0.226]]
    Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]]
    print(Demand_parameter)
    

    ## Population and quartiles for CA
    Quartile = np.genfromtxt('../Data/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)    
    Population = np.genfromtxt('../Data/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    

    ## Distance matrix for pharmacies (current)
    C_current = np.genfromtxt('../Data/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current = C_current.astype(int)
    C_current = C_current.T
    num_tracts, num_current_stores = np.shape(C_current)
   

    for Model in Model_list:

        for Chain_type in Chain_list:
        
        
            ###########################################################################

            ## Distance matrix for chain
            C_chains = np.genfromtxt('../Data/CA_dist_matrix_' + Chain_type + '.csv', delimiter = ",", dtype = float)
            C_chains = C_chains.astype(int)
            C_chains = C_chains.T
            num_tracts, num_chain_stores = np.shape(C_chains)
            C_chains = np.where(C_chains < 0, 1317574, C_chains) # High schools
            
            C_total = np.concatenate((C_current, C_chains), axis = 1)
            num_total_stores = num_current_stores + num_chain_stores
            
            C_current_walkable = np.where(C_current < 1600, 1, 0)
            C_chains_walkable = np.where(C_chains < 1600, 1, 0)
            C_total_walkable = np.where(C_total < 1600, 1, 0)
            

            ###########################################################################

            ## Demand matrix for chain

            F_DH_total = []
            for i in range(num_tracts):
            
                tract_quartile = Quartile[i]
            
                if tract_quartile == 1:
                    deltahat = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
                    tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                elif tract_quartile == 2:
                    deltahat = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
                    tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                elif tract_quartile == 3:
                    deltahat = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
                    tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                elif tract_quartile == 4:
                    deltahat = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
                    tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                else:
                    deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
                    tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
            
                F_DH_total.append(tract_willingness)
            
            F_DH_total = np.asarray(F_DH_total)
            F_DH_current = F_DH_total[:,0:num_current_stores]


            ###########################################################################

            
            for M in M_list:

                # M closest stores only
                Closest_current = np.ones((num_tracts, num_current_stores))
                Closest_total = np.ones((num_tracts, num_total_stores))
                np.put_along_axis(Closest_current, np.argpartition(C_current,M,axis=1)[:,M:],0,axis=1)
                np.put_along_axis(Closest_total, np.argpartition(C_total,M,axis=1)[:,M:],0,axis=1)
                                            
                Closest_total_chains = np.ones((num_tracts, num_chain_stores))
                np.put_along_axis(Closest_total_chains, np.argpartition(C_chains,M,axis=1)[:,M:],0,axis=1)
                Closest_total_chains = np.concatenate((Closest_current, Closest_total_chains), axis = 1)
                       
                                            
                # Complement of above matrix                                            
                Farthest_current = np.ones((num_tracts, num_current_stores)) - Closest_current
                Farthest_total_chains = np.ones((num_tracts, num_total_stores)) - Closest_total_chains

                Closest_current = Closest_current.flatten()
                Closest_total_chains = Closest_total_chains.flatten()

                Farthest_current = Farthest_current.flatten()
                Farthest_total_chains = Farthest_total_chains.flatten()
            

                for K in K_list:  
                    
                    print('Start creating summary table for ' + Chain_type + ' under Model ' + Model + ' with M = '  + str(M) + ' and K = ' + str(K) + '...\n')                    
                        
                    if Model == 'MinDist': path = '../Result/' + Model + '/' + 'M5_K' + str(K) + '/'  + Chain_type + '/'
                    else: path = '../Result/' + Model + '/' + 'M' + str(M) + '_K' + str(K) + '/'  + Chain_type + '/'


                    if Model == 'MaxRateHPIDist' or Model == 'MaxRateDist':
                        result_total_hpi = pd.read_csv(path + 'result_' + Demand_estimation + '_total.csv', delimiter = ",")
                        y_total_hpi = np.genfromtxt(path + 'y_' + Demand_estimation + '_total.csv', delimiter = ",", dtype = float)

                    else:
                        result_total_hpi = pd.read_csv(path + 'result_total.csv', delimiter = ",")
                        y_total_hpi = np.genfromtxt(path + 'y_total.csv', delimiter = ",", dtype = float)


                    y_total_hpi_closest = y_total_hpi * Closest_total_chains
                    y_total_hpi_farthest = y_total_hpi * Farthest_total_chains

                    mat_y_total_hpi = np.reshape(y_total_hpi, (num_tracts, num_total_stores))
                    mat_y_total_hpi_closest = np.reshape(y_total_hpi_closest, (num_tracts, num_total_stores))
                    mat_y_total_hpi_farthest = np.reshape(y_total_hpi_farthest, (num_tracts, num_total_stores))


                    chain_summary = create_row('Total', Model, Chain_type, M, K, Quartile, Population, mat_y_total_hpi, mat_y_total_hpi_closest, mat_y_total_hpi_farthest, result_total_hpi, F_DH_total, C_total, C_total_walkable)

                    chain_summary_table.append(chain_summary)
                    
                    
                    ###########################################################################
                    # Summary for current pharmacies

                    if Chain_type == 'Dollar': 
                        
                        if Model == 'MaxRateHPIDist' or Model == 'MaxRateDist':
                            result_current_hpi = pd.read_csv(path + 'result_' + Demand_estimation + '_current.csv', delimiter = ",")
                            y_current_hpi = np.genfromtxt(path + 'y_' + Demand_estimation + '_current.csv', delimiter = ",", dtype = float)

                        else:
                            result_current_hpi = pd.read_csv(path + 'result_current.csv', delimiter = ",")
                            y_current_hpi = np.genfromtxt(path + 'y_current.csv', delimiter = ",", dtype = float)
                        

                        y_current_hpi_closest = y_current_hpi * Closest_current
                        y_current_hpi_farthest = y_current_hpi * Farthest_current   

                        mat_y_current_hpi = np.reshape(y_current_hpi, (num_tracts, num_current_stores))
                        mat_y_current_hpi_closest = np.reshape(y_current_hpi_closest, (num_tracts, num_current_stores))
                        mat_y_current_hpi_farthest = np.reshape(y_current_hpi_farthest, (num_tracts, num_current_stores))
                 

                        chain_summary = create_row('Current', Model, Chain_type, M, K, Quartile, Population, mat_y_current_hpi, mat_y_current_hpi_closest, mat_y_current_hpi_farthest, result_current_hpi, F_DH_current, C_current, C_current_walkable)
      
                        
                        if export_tract_table==True:
                                
                                print('Start creating tract-level summaries for ' + Chain_type + ' under Model ' + Model + ' with M = '  + str(M) + ' and K = ' + str(K) + '...\n')

                                tract_summary(Model, Chain_type, M, K, Quartile, Population,\
                                mat_y_current_hpi, mat_y_current_hpi_closest, mat_y_current_hpi_farthest, result_current_hpi, F_DH_current, C_current, C_current_walkable,\
                                mat_y_total_hpi, mat_y_total_hpi_closest, mat_y_total_hpi_farthest, result_total_hpi, F_DH_total, C_total, C_total_walkable)


                        chain_summary_table.append(chain_summary)  
                    
                        
        
        chain_summary = pd.DataFrame(chain_summary_table)
        chain_summary.to_csv('../Result/summary_table_' + Demand_estimation + "_" + filename + '.csv', encoding='utf-8', index=False, header=True)
                
                


###########################################################################
###########################################################################
###########################################################################





if __name__ == "__main__":
    main()    
