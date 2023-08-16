#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
Modified from Demand/aux_datamerge.py
"""

import numpy as np
import pandas as pd
from utils.import_dist import import_dist

datadir = "/export/storage_covidvaccine/Data/"
resultdir = '/export/storage_covidvaccine/Result/'


Chain_type = 'Dollar'
M = 5
Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type, M)

capacity = 10000

tract = pd.read_csv(f'{datadir}tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int}) # tract info
block = pd.read_csv(f"{datadir}Analysis/Demand/block_data.csv") # ['zip', 'blkid', 'dist', 'population', 'weights', 'market_ids', 'nodes', 'logdist']
# block.sort_values(by=['blkid'], inplace=True)
blk_tract = pd.read_csv(f"{datadir}Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) # block-tract
block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q.csv', delimiter = ",") # block estimates
distdf = pd.read_csv(f"{datadir}Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int}) # 'blkid', 'locid', 'logdist'


### Current pharmacies
C_current = C_total[:,0:num_current_stores]
M_closest_current = np.argpartition(C_current, M, axis=1)[:,:M]
F_current = np.zeros((num_tracts, num_current_stores))

for i in range(num_tracts):

    tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
    tract_closest_sites = M_closest_current[i]

    blocks_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()

    for site in tract_closest_sites:

        tract_site_willingess = 0

        for block_id in blocks_id:
                
            block_pop = block[block.blkid == block_id].population.values[0]
            block_abd, block_distcoef = block_utils[block_utils.blkid == block_id].abd.values[0], block_utils[block_utils.blkid == block_id].distcoef.values[0]
            logdist = distdf[(distdf.blkid == block_id) & (distdf.locid == site)].logdist.values[0]

            block_utility = block_abd + block_distcoef * logdist
            tract_site_willingess += (block_pop / tract_pop) * np.exp(block_utility)/(1 + np.exp(block_utility))

        print(f'New willingness {str(tract_site_willingess)}\n')

        F_current[i][site] = tract_site_willingess


### stores
C_current = C_total[:,0:num_current_stores]
M_closest_current = np.argpartition(C_current, M, axis=1)[:,:M]
F_current = np.zeros((num_tracts, num_current_stores))

for i in range(num_tracts):

    tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
    tract_closest_sites = M_closest_current[i]

    blocks_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()

    for site in tract_closest_sites:

        tract_site_willingess = 0

        for block_id in blocks_id:
                
            block_pop = block[block.blkid == block_id].population.values[0]
            block_abd, block_distcoef = block_utils[block_utils.blkid == block_id].abd.values[0], block_utils[block_utils.blkid == block_id].distcoef.values[0]
            logdist = distdf[(distdf.blkid == block_id) & (distdf.locid == site)].logdist.values[0]

            block_utility = block_abd + block_distcoef * logdist
            tract_site_willingess += (block_pop / tract_pop) * np.exp(block_utility)/(1 + np.exp(block_utility))

        print(f'New willingness {str(tract_site_willingess)}\n')

        F_current[i][site] = tract_site_willingess





##########################################################################################






'''
## block-pharmacy distances (~500 rows per block, >1B rows)
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
#produced by /Demand/datawork/block/block_dist.py, which calls /Demand/datawork/geonear_pharmacies.do
# NOTE: the locid is some auxiliary pharmacy ID produced (as "id") at the start of block_dist.py

## block-level data
agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
agent_data_read.columns.tolist()
# the dist column is the distance to the nearest pharmacy


# add HPI quantile to agent data
hpi_level = 'tract'
if hpi_level == 'zip':
    agent_data_read = agent_data_read.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')
elif hpi_level == 'tract':
    tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") #from prep_tracts.py
    blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) #from block_cw.py
    splits = np.linspace(0, 1, nsplits+1)
    agent_data_read = agent_data_read.merge(blk_tract_cw, on='blkid', how='left')
    tract_hpi['hpi_quantile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1
    agent_data_read = agent_data_read.merge(tract_hpi[['tract', 'hpi_quantile']], on='tract', how='left')


agent_data_full = distdf.merge(agent_data_read[['blkid', 'market_ids', 'hpi_quantile']], on='blkid', how='left')

agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_bins=False, add_dummies=True, add_dist=True)
'''