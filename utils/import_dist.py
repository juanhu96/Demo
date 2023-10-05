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


def import_dist(Chain_type, M, datadir="/export/storage_covidvaccine/Data/", MAXDIST = 100000, scale_factor = 10000, nsplits = 3):

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

    Quartile = np.genfromtxt(f'{datadir}/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)
    
    ### Current ###
    C_current_mat = np.genfromtxt(f'{datadir}/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current_mat = C_current_mat.astype(int)
    C_current_mat = C_current_mat.T
    num_tracts, num_current_stores = np.shape(C_current_mat)

    ### Chains ###
    C_chains_mat = np.genfromtxt(f'{datadir}/CA_dist_matrix_' + Chain_type + '.csv', delimiter = ",", dtype = float)
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

    return Population, Quartile, p_current, p_total, pc_current, pc_total, C_total_mat, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, num_tracts, num_current_stores, num_total_stores


