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



def import_basics(Chain, M, nsplits, datadir="/export/storage_covidvaccine/Data/", MAXDIST = 100000, scale_factor = 10000):

    # ============================================================================
    # New population
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)
    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)
    
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=False, add_dist=False)
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

    # ============================================================================

    Quartile = tract_hpi['HPIQuartile']
    
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
    
    ### Travel to the closest M stores only
    Closest_current = np.ones((num_tracts, num_current_stores))
    Closest_total = np.ones((num_tracts, num_total_stores))
    np.put_along_axis(Closest_current, np.argpartition(C_current_mat,M,axis=1)[:,M:],0,axis=1)
    np.put_along_axis(Closest_total, np.argpartition(C_total_mat,M,axis=1)[:,M:],0,axis=1)
    
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

    return Population, Quartile, abd, p_current, p_total, pc_current, pc_total, C_total_mat, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, num_tracts, num_current_stores, num_total_stores




def import_BLP_estimation(Chain_type, capacity, nsplits=3, capcoef=True, heterogeneity=True, resultdir='/export/storage_covidvaccine/Result/'):

    if heterogeneity:
        if capcoef: 
            F_DH_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}_{nsplits}q_capcoefs0.csv', delimiter = ",", dtype = float) 
            F_DH_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}_{nsplits}q_capcoefs0.csv', delimiter = ",", dtype = float)
        else:
            F_DH_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}_{nsplits}q.csv', delimiter = ",", dtype = float) 
            F_DH_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}_{nsplits}q.csv', delimiter = ",", dtype = float)
    else:
        raise Exception("Warnings: homogeneous not computed")
        if capcoef: 
            F_DH_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}_{nsplits}q_capcoefs0_nodisthet.csv', delimiter = ",", dtype = float) 
            F_DH_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}_{nsplits}q_capcoefs0_nodisthet.csv', delimiter = ",", dtype = float)
        else:
            F_DH_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}_{nsplits}q_nodisthet.csv', delimiter = ",", dtype = float) 
            F_DH_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}_{nsplits}q_nodisthet.csv', delimiter = ",", dtype = float)

    F_DH_total = np.concatenate((F_DH_current, F_DH_chain), axis = 1)

    return F_DH_current, F_DH_total, F_DH_current, F_DH_total




def import_LogLin_estimation(C_total, Quartile, abd, nsplits, num_tracts, num_current_stores):

    if nsplits == 3: Demand_parameter=[[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, -0.053, -0.047, -0.039]]
    elif nsplits == 4: Demand_parameter=[[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, -0.053, -0.047, -0.039]]
    else: raise Exception("nsplits undefined, should be 3 or 4.")

    # F_D_total = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
    abd = np.nan_to_num(abd, nan=Demand_parameter[0][0])
    F_D_total = abd.reshape(8057, 1) + Demand_parameter[0][1] * np.log(C_total/1000)
    F_D_current = F_D_total[:,0:num_current_stores]

    F_DH_total = []

    for i in range(num_tracts):
                
        tract_quartile = Quartile[i]
                
        if tract_quartile == 1:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 2:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 3:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 4:
            # if nsplits = 3, this would never show up
            # TODO: need to change the order of the demand parameters a a bit
            tract_willingness = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
        else:
            tract_willingness = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
                
        F_DH_total.append(tract_willingness)
                
    F_DH_total = np.asarray(F_DH_total)
    F_DH_current = F_DH_total[:,0:num_current_stores]

    return F_D_current, F_D_total, F_DH_current, F_DH_total