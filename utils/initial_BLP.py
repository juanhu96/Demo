#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
- Import BLP demand estimation and compute F
- Check demand estimation (tract/zip level)
"""

import time
import numpy as np
import pandas as pd
import random


def initial_BLP_estimation(capacity, groups, capcoef, mnl, flexible_consideration, logdist_above, logdist_above_thresh, setting_tag, datadir='/export/storage_covidvaccine/Demo/Data/', resultdir='/export/storage_covidvaccine/Demo/Result/'):
    
    '''
    tract_centroids.csv : tract info
    block_data.csv : block info (needs to be sorted), ['zip', 'blkid', 'dist', 'population', 'weights', 'market_ids', 'nodes', 'logdist']
    blk_tract.csv : block-tract pairs
    agent_results_{capacity}_200_3q.csv : block-level demand estimates, based on capacity K = {8000, 10000, 12000, 15000}
    ca_blk_pharm_dist.csv : block-pharmacy
    ca_blk_{Chain_type}_dist.csv : block-chain, need to be precomputed from block_dist_chain.py (with geonear in STATA)
    '''
    
    print(f'Start initializing BLP matrices from estimation under capacity {str(capacity)} with {groups} groups, with setting tag {setting_tag}\n')

    areas = pd.read_csv(f"{datadir}/../areas.csv")
    areas.rename(columns={'ZIP': 'zip'}, inplace=True)
    locations = pd.read_csv(f"{datadir}/../locations.csv")

    ### Tract-block
    tract = pd.read_csv(f'{datadir}/Raw/tract/tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int})
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv') 
    block.sort_values(by=['blkid'], inplace=True)
    blk_tract = pd.read_csv(f'{datadir}/Intermediate/blk_ziptract.csv')
    block_utils = pd.read_csv(f'{resultdir}Demand/agent_results{setting_tag}.csv', delimiter = ",")

    area_blk_tract = blk_tract[blk_tract.zip.isin(areas.zip)]
    tract_ids = area_blk_tract.tract.unique()
    tract_ids.sort()
    tract = tract.loc[tract.GEOID.isin(tract_ids)]
    tract.reset_index(drop=True, inplace=True)
    print(f'{len(tract_ids)} tracts in this case')

    ### Distance pairs
    distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_current_dist.csv', dtype={'locid': int, 'blkid': int})
    C_total, num_tracts, num_current_stores, num_total_stores = import_dist(tract, locations)

    construct_F_BLP(capacity, C_total, num_tracts, num_total_stores, tract, block, blk_tract, block_utils, distdf, logdist_above, logdist_above_thresh, setting_tag, M = 5)

    return



def import_dist(tract, locations, datadir="/export/storage_covidvaccine/Demo/Data"):

    tract_latlong = tract[['LATITUDE', 'LONGITUDE']].values
    locations_latlong = locations[['latitude', 'longitude']].values

    # 1 degree of latitude is approximately 111 kilometers.
    # 1 degree of longitude is approximately 87.56 km in CA (average latitude of 38°).
    # The average of these two values is 99.28 km.
    # geodesic gives a more accurate distance than the above approximation.
    # 0-based index
    distances = np.sqrt(((tract_latlong[:, np.newaxis, :] - locations_latlong[np.newaxis, :, :]) ** 2).sum(axis=2))
    distances.shape
    distances = distances * 99.28
    C_total_mat = distances.astype(int)
    num_tracts, num_total_stores = np.shape(C_total_mat)
    num_current_stores = locations['open'].sum()

    C_total_df = pd.DataFrame(C_total_mat)
    C_total_df.to_csv(f'{datadir}/Intermediate/C_total_mat.csv', header=False, index=False)

    return C_total_mat, num_tracts, num_current_stores, num_total_stores



def construct_F_BLP(capacity, C_total, num_tracts, num_total_stores, tract, block, blk_tract, block_utils, distdf, logdist_above, logdist_above_thresh,
setting_tag, M=100, resultdir='/export/storage_covidvaccine/Demo/Result/'):
    
    '''
    M matter here because consideration set C could go up to 300
    But setting M = 100 covers 99 quantile of the tracts, and the median is only 5
    '''
    
    # ========================================================================================
    # CURRENT LOCATIONS
        
    print('Constructing F_BLP_total...\n')

    M_closest_total = np.argpartition(C_total, M, axis=1)[:,:M]
    V_total = np.zeros((num_tracts, num_total_stores))

    start = time.time()

    for i in range(num_tracts):
    # for i in range(1):
        tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
        blk_tract_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()
        block_id = block[block.blkid.isin(blk_tract_id)].blkid.to_list()
        block_utils_id = block_utils[block_utils.blkid.isin(blk_tract_id)].blkid.to_list()

        if len(blk_tract_id) != len(block_id) or len(block_id) != len(block_utils_id) or len(block_utils_id) != len(blk_tract_id):
            common_blocks_id = set(blk_tract_id) & set(block_id) & set(block_utils_id)
        else:
            common_blocks_id = blk_tract_id

        blocks_pop = block[block.blkid.isin(common_blocks_id)].population.to_numpy()
        blocks_abd = block_utils[block_utils.blkid.isin(common_blocks_id)].abd.to_numpy()
        blocks_distcoef = block_utils[block_utils.blkid.isin(common_blocks_id)].distcoef.to_numpy()
        block_distdf = distdf[distdf.blkid.isin(common_blocks_id)]

        tract_pop = np.sum(blocks_pop)

        for index, site in enumerate(M_closest_total[i]):

            tract_site_willingness = 0    
            dist_blocks_id = block_distdf[block_distdf.locid == site].blkid.to_list()

            if len(dist_blocks_id) != len(common_blocks_id):

                # take ths subset
                temp_common_blocks_id = set(dist_blocks_id) & set(common_blocks_id)
                temp_blocks_pop = block[block.blkid.isin(temp_common_blocks_id)].population.to_numpy()
                temp_blocks_abd = block_utils[block_utils.blkid.isin(temp_common_blocks_id)].abd.to_numpy()
                temp_blocks_distcoef = block_utils[block_utils.blkid.isin(temp_common_blocks_id)].distcoef.to_numpy()
                temp_block_distdf = block_distdf[block_distdf.blkid.isin(temp_common_blocks_id)]
                temp_tract_pop = np.sum(temp_blocks_pop)
                
                logdists = temp_block_distdf[temp_block_distdf.locid == site].logdist.to_numpy()
                if logdist_above: 
                    # logdists = np.maximum(logdists, np.log(logdist_above_thresh))  
                    dists = temp_block_distdf[temp_block_distdf.locid == site].dist.to_numpy()
                    logdists = np.log(np.maximum(dists + 1 - logdist_above_thresh, 1))

                blocks_utility = temp_blocks_abd + temp_blocks_distcoef * logdists
                # tract_site_willingness = np.sum((temp_blocks_pop / temp_tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))
                tract_site_preference = np.exp(np.sum((temp_blocks_pop / temp_tract_pop) * blocks_utility))

            else:
                
                logdists = block_distdf[block_distdf.locid == site].logdist.to_numpy()
                if logdist_above: 
                    # logdists = np.maximum(logdists, np.log(logdist_above_thresh))
                    dists = block_distdf[block_distdf.locid == site].dist.to_numpy()
                    logdists = np.log(np.maximum(dists + 1 - logdist_above_thresh, 1))

                blocks_utility = blocks_abd + blocks_distcoef * logdists
                # tract_site_willingness = np.sum((blocks_pop / tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))
                tract_site_preference = np.exp(np.sum((blocks_pop / tract_pop) * blocks_utility))

            # F_total[i][site] = tract_site_willingness
            V_total[i][site] = tract_site_preference # a numerical easier approx for tract utility

    print(f'Finished computing, time spent: {str(int(time.time()-start))}; Start exporting...\n')

    # F_total_df = pd.DataFrame(F_total)
    V_total_df = pd.DataFrame(V_total)

    # F_total_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_total{setting_tag}.csv', header=False, index=False)
    V_total_df.to_csv(f'{resultdir}BLP_matrix/V_total{setting_tag}.csv', header=False, index=False)

    return
