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



def import_basics(M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor, datadir="/export/storage_covidvaccine/Demo/Data/", MAXDIST=100000):

    # ============================================================================
    # New population
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)
    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)
    
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))

    block = block.loc[block.market_ids.isin(mkts_in_both), :]
    df = df.loc[df.market_ids.isin(mkts_in_both), :]
    block = block.merge(df[['market_ids', 'RC']], on='market_ids', how='left')
    splits = np.linspace(0, 1, nsplits+1)

    areas = pd.read_csv(f"{datadir}/../areas.csv")
    areas.rename(columns={'ZIP': 'zip'}, inplace=True)
    tract = pd.read_csv(f'{datadir}/Raw/tract/tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int})
    blk_tract = pd.read_csv(f'{datadir}/Intermediate/blk_ziptract.csv')
    area_blk_tract = blk_tract[blk_tract.zip.isin(areas.zip)]
    tract_ids = area_blk_tract.tract.unique()
    tract_ids.sort()
    tract = tract.loc[tract.GEOID.isin(tract_ids)]
    tract.reset_index(drop=True, inplace=True)
    tract.rename(columns={'GEOID': 'tract'}, inplace=True)

    temp = block.merge(blk_tract, on='blkid', how='left')
    blk_tract_pop = temp.groupby('tract')['population'].sum().reset_index()
    tract = tract.merge(blk_tract_pop[['tract','population']], on='tract', how='left')
    Population = tract['population'].astype(int)


    ### Current ###
    C_total_mat = np.genfromtxt(f'{datadir}/Intermediate/C_total_mat.csv', delimiter = ",", dtype = float)
    C_total_mat = C_total_mat.astype(int)
    num_tracts, num_total_stores = np.shape(C_total_mat)
    print(C_total_mat.shape)

    locations = pd.read_csv(f"{datadir}/../locations.csv")
    num_current_stores = locations['open'].sum()
    open_locations = locations['open'] == 1  # This creates a boolean mask for open locations
    C_current_mat = C_total_mat[:, open_locations]
    print(C_current_mat.shape)


    ###########################################################################

    print(f'Closest_total and C follows fix rank of {M}\n')
    Closest_current = np.ones((num_tracts, num_current_stores))
    Closest_total = np.ones((num_tracts, num_total_stores))
    np.put_along_axis(Closest_current, np.argpartition(C_current_mat,M,axis=1)[:,M:],0,axis=1)
    np.put_along_axis(Closest_total, np.argpartition(C_total_mat,M,axis=1)[:,M:],0,axis=1)
    C = np.argsort(C_total_mat, axis=1)[:, :M]
    # print("========= USING THE TEN CLOSEST INSTEAD ============\n")
    # C = np.argsort(C_total_mat, axis=1)[:, :10] # test for evaluation issue

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

    return locations, Population, p_current, p_total, pc_current, pc_total, C_total_mat, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, C, num_tracts, num_current_stores, num_total_stores




def import_estimation(Model_name, R, A, random_seed, setting_tag, resultdir='/export/storage_covidvaccine/Demo/Result/'):
    
    if R is not None: setting_tag = setting_tag.replace(f'_R{R}', '')
    if A is not None: setting_tag = setting_tag.replace(f'_A{A}', '')
    if random_seed is not None: setting_tag = setting_tag.replace(f'_randomseed{random_seed}', '')

    print(f"Importing estimation from {Model_name}{setting_tag}.csv\n")
    F_total = np.genfromtxt(f'{resultdir}BLP_matrix/{Model_name}_total{setting_tag}.csv', delimiter = ",", dtype = float) 

    return F_total
