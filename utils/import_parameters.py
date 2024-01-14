#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import pandas as pd
import numpy as np

try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de



def import_basics(Chain, M, nsplits, flexible_consideration, datadir="/export/storage_covidvaccine/Data/", MAXDIST = 100000, scale_factor = 10000):

    # ============================================================================
    # New population
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)
    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)
    
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = de.hpi_dist_terms(df, nsplits=nsplits, add_hpi_bins=True, add_hpi_dummies=False, add_dist=False)
    df_temp = df.copy()
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))

    block = block.loc[block.market_ids.isin(mkts_in_both), :]
    df = df.loc[df.market_ids.isin(mkts_in_both), :]
    block = block.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')
    tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") # 8057 tracts
    splits = np.linspace(0, 1, nsplits+1)
    tract_hpi['HPIQuartile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1

    tract_hpi['Raw_Population'] = np.genfromtxt(f'{datadir}/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid'])
    temp = block.merge(blk_tract_cw, on='blkid', how='left')
    blk_tract_pop = temp.groupby('tract')['population'].sum().reset_index() # only 8021
    tract_hpi = tract_hpi.merge(blk_tract_pop[['tract','population']], on='tract', how='left')
    tract_hpi['population'].fillna(tract_hpi['Raw_Population'], inplace=True)
    Population = tract_hpi['population'].astype(int)

    tract_hpi['Popdensity'] = np.genfromtxt(f'{datadir}/CA_density.csv', delimiter = ",", dtype = int)
    tract_hpi['Popdensity_group'] = pd.cut(tract_hpi['Popdensity'], bins=[0, 1000, 3000, np.inf], labels=['rural', 'suburban', 'urban'], right=False)

    # ============================================================================

    Quartile = tract_hpi['HPIQuartile']
    Popdensity = tract_hpi['Popdensity_group']
    
    # For LogLinear only
    tract_abd = pd.read_csv(f"{datadir}/Intermediate/tract_abd.csv", usecols=['tract', 'abd'])
    abd = tract_abd['abd'].values
    
    ### Current ###
    C_current_mat = np.genfromtxt(f'{datadir}/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current_mat = C_current_mat.astype(int)
    C_current_mat = C_current_mat.T
    num_tracts, num_current_stores = np.shape(C_current_mat)

    ### Chains ###
    C_chains_mat = np.genfromtxt(f'{datadir}/CA_dist_matrix_{Chain}.csv', delimiter = ",", dtype = float)
    C_chains_mat = C_chains_mat.astype(int)
    C_chains_mat = C_chains_mat.T
    num_tracts, num_chains_stores = np.shape(C_chains_mat)
    C_chains_mat = np.where(C_chains_mat < 0, 1317574, C_chains_mat) # avoid negative numbers for high schools
    
    ### Total ###
    C_total_mat = np.concatenate((C_current_mat, C_chains_mat), axis = 1)
    num_total_stores = num_current_stores + num_chains_stores
    ###########################################################################

    consideration_case = 'flexible' if flexible_consideration else 'fix_rank'

    if consideration_case == 'fix_rank':
        print(f'fix_rank of {M}\n')
        Closest_current = np.ones((num_tracts, num_current_stores))
        Closest_total = np.ones((num_tracts, num_total_stores))
        np.put_along_axis(Closest_current, np.argpartition(C_current_mat,M,axis=1)[:,M:],0,axis=1)
        np.put_along_axis(Closest_total, np.argpartition(C_total_mat,M,axis=1)[:,M:],0,axis=1)
        C = np.argsort(C_total_mat, axis=1)[:, :M]

    elif consideration_case == 'fix_dist':

        consideration_dist = 3200 # approximately 2 miles
        mask_current = C_current_mat < consideration_dist
        mask_total = C_total_mat < consideration_dist

        rows_without_lower = np.all(mask_current == False, axis=1)
        mask_current[rows_without_lower, np.argmin(C_current_mat[rows_without_lower], axis=1)] = True

        rows_without_lower = np.all(mask_total == False, axis=1)
        mask_total[rows_without_lower, np.argmin(C_total_mat[rows_without_lower], axis=1)] = True

        Closest_current = mask_current.astype(int)
        Closest_total = mask_total.astype(int)

        Consideration_set = np.sum(Closest_total, axis = 1)

        C = []
        for row in range(Closest_total.shape[0]):
            indices = np.where(Closest_total[row] == 1)[0].tolist()
            C.append(indices)

    elif consideration_case == 'flexible':
        print("flexible consideration set\n")
        D_values = {'urban': 2000, 'suburban': 3000, 'rural': 15000}
        D = np.array([D_values[density_group] for density_group in Popdensity])

        Closest_current = np.zeros_like(C_current_mat)
        Closest_total = np.zeros_like(C_total_mat)

        for (current_row, total_row, d, i) in zip(C_current_mat, C_total_mat, D, range(len(D))):

            current_mask = current_row < d
            if np.any(current_mask):
                Closest_current[i, current_mask] = 1
            else:
                Closest_current[i, np.argmin(current_row)] = 1

            total_mask = total_row < d
            if np.any(total_mask):
                Closest_total[i, total_mask] = 1
            else:
                Closest_total[i, np.argmin(total_row)] = 1

        C = []
        for row in range(Closest_total.shape[0]):
            indices = np.where(Closest_total[row] == 1)[0].tolist()
            C.append(indices)


    # def summary_consideration_set(C):

    #     lengths = [len(sublist) for sublist in C]
    #     quantile_50 = np.quantile(lengths, 0.50)
    #     quantile_75 = np.quantile(lengths, 0.75)
    #     quantile_90 = np.quantile(lengths, 0.90)
    #     quantile_95 = np.quantile(lengths, 0.95)
    #     quantile_97 = np.quantile(lengths, 0.97)
    #     quantile_98 = np.quantile(lengths, 0.98)
    #     quantile_99 = np.quantile(lengths, 0.99)
    #     min_length = min(lengths)
    #     max_length = max(lengths)

    #     print(min_length, max_length, quantile_50, quantile_75, quantile_90, quantile_95, quantile_97, quantile_98, quantile_99)

    # summary_consideration_set(C)

    ###########################################################################

    C_currentMinDist = C_current_mat * Closest_current
    C_totalMinDist = C_total_mat * Closest_total
    C_currentMinDist = np.where(C_currentMinDist == 0, MAXDIST, C_currentMinDist)
    C_totalMinDist = np.where(C_totalMinDist == 0, MAXDIST, C_totalMinDist)

    ###########################################################################

    C_current = C_current_mat.flatten() / scale_factor
    C_total = C_total_mat.flatten() / scale_factor

    Closest_current = Closest_current.flatten()
    Closest_total = Closest_total.flatten()

    c_currentMinDist = C_currentMinDist.flatten() / scale_factor
    c_totalMinDist = C_totalMinDist.flatten() / scale_factor

    ###########################################################################

    # n copies of demand
    p_total = np.tile(Population, num_total_stores)
    p_total = np.reshape(p_total, (num_total_stores, num_tracts))
    p_total = p_total.T.flatten()
       
    p_current = np.tile(Population, num_current_stores)
    p_current = np.reshape(p_current, (num_current_stores, num_tracts))
    p_current = p_current.T.flatten()
    
    # population * distance 
    pc_current = p_current * C_current
    pc_total = p_total * C_total

    return Population, Quartile, abd, p_current, p_total, pc_current, pc_total, C_total_mat, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, C, num_tracts, num_current_stores, num_total_stores




def import_BLP_estimation(Chain_type, setting_tag, resultdir='/export/storage_covidvaccine/Result/'):

    print(f"import MNL estimation from file BLP_current{setting_tag}\n")

    F_DH_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current{setting_tag}.csv', delimiter = ",", dtype = float) 
    F_DH_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}{setting_tag}.csv', delimiter = ",", dtype = float)
    F_DH_total = np.concatenate((F_DH_current, F_DH_chain), axis = 1)

    return F_DH_current, F_DH_total, F_DH_current, F_DH_total




def import_LogLin_estimation(C_total, abd, num_current_stores):

    Demand_parameter=[[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, -0.053, -0.047, -0.039]]

    # F_D_total = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
    abd = np.nan_to_num(abd, nan=Demand_parameter[0][0])
    F_D_total = abd.reshape(8057, 1) + Demand_parameter[0][1] * np.log(C_total/1000)
    F_D_current = F_D_total[:,0:num_current_stores]

    return F_D_current, F_D_total, F_D_current, F_D_total



def import_MNL_estimation(Chain, setting_tag, resultdir='/export/storage_covidvaccine/Result/'):

    print(f"import MNL estimation from file V_current{setting_tag}\n")

    V_current = np.genfromtxt(f'{resultdir}BLP_matrix/V_current{setting_tag}.csv', delimiter = ",", dtype = float) 
    V_chain = np.genfromtxt(f'{resultdir}BLP_matrix/V_{Chain}{setting_tag}.csv', delimiter = ",", dtype = float)
    V_total = np.concatenate((V_current, V_chain), axis = 1)
    
    return V_current, V_total