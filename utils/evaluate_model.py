#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21, 2022
@Author: Jingyuan Hu 
"""
import os
import pandas as pd
import numpy as np
import subprocess
import geopandas as gpd

import time
import gurobipy as gp
from gurobipy import GRB, quicksum

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


scale_factor = 10000

try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de



def compute_distdf(z: np.ndarray,
                   setting_tag: str,
                   path: str,
                   datadir: str = '/export/storage_covidvaccine/Demo/Data',
                   resultdir: str = '/export/storage_covidvaccine/Demo/Result',
                   within: int = 3000,
                   limit: int = 50):
    
    print(f"Distdf not computed for current setting, start computing; File saved as ca_blk_selected_dist_total{setting_tag}.csv\n")

    all_locations = pd.read_csv(f"{datadir}/../locations.csv")
    selected_locations = all_locations[z == 1]
    selected_locations['id'] = range(selected_locations.shape[0])

    # ============================= STATA ==================================

    chainlocpath = f"{path}/ca_selected_locations_total{setting_tag}.dta"
    selected_locations.to_stata(chainlocpath, write_index=False)

    baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
    chain = pd.read_stata(chainlocpath)
    chain.tail()
    len(set(chain.id))

    outpath = f"{path}/ca_blk_selected_dist_total{setting_tag}.csv"
    within = 3000 # km
    limit = 50 # number of chain stores to consider

    # os.chdir("../output_log/Stata/")
    output = subprocess.run(["stata-mp", "-b", "do", f"/mnt/phd/jihu/Demo/Demand/datawork/geonear_current.do", baselocpath, chainlocpath, outpath, str(within), str(limit)], capture_output=True, text=True)
    print(output.stdout)
    print(output.stderr)

    # ============================= STATA ==================================

    return 



# ===========================================================================


def construct_blocks(M, K, nsplits, flexible_consideration, flex_thresh, R, A, setting_tag, path, random_seed=None, Pharmacy=False, 
datadir='/export/storage_covidvaccine/Demo/Data', resultdir='/export/storage_covidvaccine/Demo/Result'):

    # Block basic info
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)
    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)

    # Distance pairs
    distdf = pd.read_csv(f'{path}/ca_blk_selected_dist_total{setting_tag}.csv', dtype={'locid': int, 'blkid': int})

    # Block estimation
    temp_setting_tag = setting_tag.replace('_norandomterm', '') # estimation independent of random term
    temp_setting_tag = temp_setting_tag.replace('_loglintemp', '') # estimation independent of loglin form
    if R is not None: temp_setting_tag = temp_setting_tag.replace(f'_R{R}', '')
    if A is not None: temp_setting_tag = temp_setting_tag.replace(f'_A{A}', '')
    if random_seed is not None: temp_setting_tag = temp_setting_tag.replace(f'_randomseed{random_seed}', '')

    block_utils = pd.read_csv(f'{resultdir}/Demand/agent_results{temp_setting_tag}.csv', delimiter = ",")
    block_utils = block_utils.loc[block_utils.blkid.isin(blocks_unique), :]


    # Keep markets in both
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))

    distdf = distdf.groupby('blkid').head(M).reset_index(drop=True)
    distdf = distdf.loc[distdf.blkid.isin(blocks_unique), :]

    # Subset blocks and add HPI
    block = block.loc[block.market_ids.isin(mkts_in_both), :]
    df = df.loc[df.market_ids.isin(mkts_in_both), :]
    distdf = distdf.loc[distdf.blkid.isin(block.blkid.unique()), :]
    block = block.merge(df[['market_ids', 'RC']], on='market_ids', how='left')
    
    return block, block_utils, distdf


# ===========================================================================


def run_assignment(M, K, nsplits, capcoef, mnl, setting_tag, block, block_utils, distdf, path, Pharmacy=False):

    print('Start assignments...\n')

    distcoefs = block_utils.distcoef.values
    abd = block_utils.abd.values

    dist_grp = distdf.groupby('blkid')
    locs = dist_grp.locid.apply(np.array).values # list of lists of location IDs, corresponding to dists
    dists = dist_grp.logdist.apply(np.array).values # list of lists of distances, sorted ascending
    geog_pops = block.population.values # identical to cw_pop.population.values done by Li
    geog_pops = np.array(geog_pops).astype(int).tolist() # updated: float to int
    
    # ===========================================================================

    economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=M, mnl=mnl)
    af.random_fcfs(economy, distcoefs, abd, K, mnl=mnl, evaluation=True)
    assignment = economy.assignments

    # ===========================================================================

    # For flexible_consideration: the length of each array is different, pad each array with zeros make the output so large

    def pad_results(arr_to_pad):
        max_length = max(len(arr) for arr in arr_to_pad)
        arr_padded = np.array([np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in arr_to_pad])
        return arr_padded

    def pad_results_sparse(arr_to_pad):
        # computationally effective
        from scipy import sparse

        max_length = max(len(arr) for arr in arr_to_pad)
        rows = len(arr_to_pad)
        data = [(i, j, val) for i, arr in enumerate(arr_to_pad) for j, val in enumerate(arr)]
        row_indices, col_indices, values = zip(*data)
        arr_sparse = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(rows, max_length))

        return arr_sparse

    locs_padded_sparse = pad_results_sparse(locs)
    dists_padded_sparse = pad_results_sparse(dists)
    assignment_padded_sparse = pad_results_sparse(assignment)

    locs_padded_dense = locs_padded_sparse.toarray()
    dists_padded_dense = dists_padded_sparse.toarray()
    assignment_padded_dense = assignment_padded_sparse.toarray()

    assignment = np.array(assignment_padded_dense)
    locs = np.stack(locs_padded_dense, axis=0)
    dists = np.stack(dists_padded_dense, axis=0)

    np.savetxt(f'{path}/locs{setting_tag}.csv', locs, fmt='%s')
    np.savetxt(f'{path}/dists{setting_tag}.csv', dists, fmt='%s')
    np.savetxt(f'{path}/assignment{setting_tag}.csv', assignment, fmt='%s')
        
    return assignment, locs, dists


# ===========================================================================

def summary_statistics(assignment, locs, dists, block, setting_tag, path, datadir="/export/storage_covidvaccine/Demo/Data/"):
    
    total_population = sum(block.population)

    assignment_sums = np.sum(assignment, axis=1)
    block['assignments'] = assignment_sums

    zip_results = block.groupby('market_ids').agg({'assignments': 'sum', 'population': 'sum'}).reset_index()
    zip_results['Rates_after'] = zip_results['assignments'] / zip_results['population']

    areas = pd.read_csv(f"{datadir}/../areas.csv")
    areas.rename(columns={'ZIP': 'zip'}, inplace=True)
    areas = areas.merge(zip_results, left_on='zip', right_on='market_ids', how='left')
    areas.to_csv(f"{path}/areas{setting_tag}.csv", index=False)

    return