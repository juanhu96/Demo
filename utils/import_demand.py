#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
Import BLP demand estimation and compute F
"""

import time
import numpy as np
import pandas as pd



def initial_BLP_estimation(Chain_type, capacity, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):
    
    '''
    tract_centroids.csv : tract info
    block_data.csv : block info (needs to be sorted), ['zip', 'blkid', 'dist', 'population', 'weights', 'market_ids', 'nodes', 'logdist']
    blk_tract.csv : block-tract pairs
    agent_results_{capacity}_200_3q.csv : block-level demand estimates, based on capacity K = {8000, 10000, 12000, 15000}
    ca_blk_pharm_dist.csv : block-pharmacy
    ca_blk_{Chain_type}_dist.csv : block-chain, need to be computed from block_dist_chain.py (with geonear in STATA)
    '''

    tract = pd.read_csv(f'{datadir}tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int})
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv') 
    block.sort_values(by=['blkid'], inplace=True)
    blk_tract = pd.read_csv(f'{datadir}/Intermediate/blk_tract.csv', usecols=['tract', 'blkid']) 
    block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q.csv', delimiter = ",") 

    ### Distance pairs
    distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
    distdf_chain = pd.read_csv(f'{datadir}Intermediate/ca_blk_{Chain_type}_dist.csv', dtype={'locid': int, 'blkid': int}) # 

    C_total, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type=Chain_type, M=20)
    construct_F_BLP(Chain_type, capacity, C_total, num_tracts, num_current_stores, num_total_stores, tract, block, blk_tract, block_utils, distdf, distdf_chain)



def import_dist(Chain_type, M, datadir="/export/storage_covidvaccine/Data"):

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

    return C_total_mat, num_tracts, num_current_stores, num_total_stores



def construct_F_BLP(Chain_type, capacity, C_total, num_tracts, num_current_stores, num_total_stores, tract, block, blk_tract, block_utils, distdf, distdf_chain, M=10, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):
    
    '''
    M here doesn't matter, because we have C_closest in the optimization constraint
    (as long as greater than 10)
    '''

    if Chain_type == 'Dollar':
        
        print('Constructing F_BLP_current...\n')

        C_current = C_total[:,0:num_current_stores]
        M_closest_current = np.argpartition(C_current, M, axis=1)[:,:M]
        F_current = np.zeros((num_tracts, num_current_stores))

        start = time.time()

        for i in range(num_tracts):

            tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]

            blocks_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()
            blocks_pop = block[block.blkid.isin(blocks_id)].population.to_numpy()
            blocks_abd = block_utils[block_utils.blkid.isin(blocks_id)].abd.to_numpy()
            blocks_distcoef = block_utils[block_utils.blkid.isin(blocks_id)].distcoef.to_numpy()
            block_distdf = distdf[distdf.blkid.isin(blocks_id)]

            tract_pop = np.sum(blocks_pop)

            for site in M_closest_current[i]:
                
                tract_site_willingess = 0            
                logdists = block_distdf[block_distdf.locid == site].logdist.to_numpy()

                blocks_utility = blocks_abd + blocks_distcoef * logdists
                tract_site_willingess = np.sum((blocks_pop / tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))

                # for block_id in blocks_id:
                        
                #     block_pop = block[block.blkid == block_id].population.values[0]
                #     block_abd, block_distcoef = block_utils[block_utils.blkid == block_id].abd.values[0], block_utils[block_utils.blkid == block_id].distcoef.values[0]
                #     logdist = distdf[(distdf.blkid == block_id) & (distdf.locid == site)].logdist.values[0]

                #     block_utility = block_abd + block_distcoef * logdist
                #     tract_site_willingess += (block_pop / tract_pop) * (np.exp(block_utility)/(1 + np.exp(block_utility)))

                F_current[i][site] = tract_site_willingess

        print(f'Finished computing, time spent: {str(int(time.time()-start))}; Start exporting...\n')
        F_current_df = pd.DataFrame(F_current)
        F_current_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}.csv', header=False, index=False)


    ######################################################################
    
    print(f'Constructing F_BLP_{Chain_type}...\n')

    C_chains = C_total[:,num_current_stores:num_total_stores]
    M_closest_chains = np.argpartition(C_chains, M, axis=1)[:,:M]
    F_chains = np.zeros((num_tracts, num_total_stores-num_current_stores))

    start = time.time()

    for i in range(num_tracts):

        tract_id = tract['GEOID'][i]

        blocks_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()
        blocks_pop = block[block.blkid.isin(blocks_id)].population.to_numpy()
        blocks_abd = block_utils[block_utils.blkid.isin(blocks_id)].abd.to_numpy()
        blocks_distcoef = block_utils[block_utils.blkid.isin(blocks_id)].distcoef.to_numpy()

        block_distdf = distdf_chain[distdf_chain.blkid.isin(blocks_id)] # only difference

        tract_pop = np.sum(blocks_pop)

        for site in M_closest_chains[i]:

            tract_site_willingess = 0            
            logdists = block_distdf[block_distdf.locid == site].logdist.to_numpy()

            blocks_utility = blocks_abd + blocks_distcoef * logdists
            tract_site_willingess = np.sum((blocks_pop / tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))
            
            F_chains[i][site] = tract_site_willingess

    print(f'Finished computing, time spent: {str(int(time.time()-start))}; Start exporting...\n')    
    F_chains_df = pd.DataFrame(F_chains)
    F_chains_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}.csv', header=False, index=False)



def import_BLP_estimation(Chain_type, capacity, resultdir='/export/storage_covidvaccine/Result/'):

    F_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}.csv', delimiter = ",", dtype = float) 
    F_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}.csv', delimiter = ",", dtype = float)
    F_total = np.concatenate((F_current, F_chain), axis = 1)

    return F_current, F_total, F_current, F_total