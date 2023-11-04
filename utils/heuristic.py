#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2023
@Author: Jingyuan Hu 
"""

import pandas as pd
import numpy as np
from utils.partnerships_summary_helpers import import_dataset, import_locations, import_solution
from utils.evaluate_model import construct_blocks

try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de
    

def rescale_estimation(F_DH_total, Model, Chain, M, K, nsplits, capcoef, R, constraint, datadir='/export/storage_covidvaccine/Data', resultdir='/export/storage_covidvaccine/Result'):

    '''
    Scale F_BLP based on our previous solution from optimization and evaluation

    block, blk_tract_cw, locs, dists, assignment are all same size
    locs is 0-indexed


    A = np.array([0, 0.5, 0.8])
    B = np.array([0, 0, 1])

    print(np.nan_to_num(A / B, posinf=0))
    print((np.nan_to_num(A / B, posinf=0) + 1) / 2)
    print(np.multiply(B, (np.nan_to_num(A / B, posinf=0) + 1) / 2))
    '''

    _, df_temp, block, tract_hpi, blk_tract_cw = import_dataset(nsplits, datadir)
    _, _, num_tracts, num_current_stores, num_total_stores, _, _, _, _ = import_locations(df_temp, Chain)

    if capcoef: path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef/{Chain}/'
    else: path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q/{Chain}/'

    block, block_utils = import_blocks(Chain, M, K, nsplits, R, True, constraint, path)

    path = path + constraint + '/'
    if R is not None: z, mat_y, _, locs, dists, assignment = import_solution(path, Chain, K, num_tracts, num_total_stores, num_current_stores, constraint, R)
    else: z, mat_y, _, locs, dists, assignment = import_solution(path, Chain, K, num_tracts, num_total_stores, num_current_stores, constraint)

    tract = pd.read_csv(f'{datadir}/tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int})
    block_tract = block.merge(blk_tract_cw, on='blkid', how='left') # block_tract.blkid.nunique()

    # ==============================================================================================

    # tract-level vaccination (random FCFS)
    
    actual_assignment = np.zeros((num_tracts, num_total_stores))
    for i in range(num_tracts):
        
        tract_id, tract_pop = tract_hpi['tract'][i], tract_hpi['population'][i]
        blk_tract_id = blk_tract_cw[blk_tract_cw.tract == tract_id].blkid.to_list() # blocks in tract
        
        mask = np.isin(block.blkid, blk_tract_id)
        tract_locs, tract_assignment = locs[mask], assignment[mask]
        flat_tract_locs, flat_tract_assignment = tract_locs.flatten(), tract_assignment.flatten()
        unique_locs = np.unique(flat_tract_locs)

        for unique_loc in unique_locs:
            actual_assignment[i][int(unique_loc)] = flat_tract_assignment[flat_tract_locs == unique_loc].sum() / tract_pop
    
    # ==============================================================================================

    # tract-level vaccination (full control)

    opt_assignment = np.multiply(F_DH_total, mat_y)

    # rescale
    prop_assignment = (np.nan_to_num(actual_assignment / opt_assignment, posinf=0) + 1) / 2 # e.g., 0.5 + 1 / 2 = 0.75
    # print(np.min(prop_assignment), np.max(prop_assignment), np.unravel_index(np.argmax(prop_assignment), prop_assignment.shape))

    F_DH_total_updated = np.multiply(F_DH_total, prop_assignment)
    # NOTE: e.g., for (2752, 22), the actual vaccination is 0.10 and optimization vaccination is 2e-16
    F_DH_total_updated[F_DH_total_updated > 1] = 1 
    # print(np.min(F_DH_total_updated), np.max(F_DH_total_updated), np.unravel_index(np.argmax(F_DH_total_updated), F_DH_total_updated.shape))

    return F_DH_total_updated




def import_blocks(Chain, M, K, nsplits, R, heuristic, constraint, path, datadir='/export/storage_covidvaccine/Data', resultdir='/export/storage_covidvaccine/Result'):

    print('Start importing blocks...\n')
    if constraint != 'None': path = f'{path}/{constraint}'

    # Block basic info
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)

    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)


    # Block estimation
    block_utils = pd.read_csv(f'{resultdir}/Demand/agent_results_{K}_200_3q.csv', delimiter = ",") 
    block_utils = block_utils.loc[block_utils.blkid.isin(blocks_unique), :]


    # Keep markets in both
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=False, add_dist=False)
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))


    # Subset blocks and add HPI
    block = block.loc[block.market_ids.isin(mkts_in_both), :]
    df = df.loc[df.market_ids.isin(mkts_in_both), :]
    block = block.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')


    return block, block_utils