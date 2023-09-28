#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
Import BLP demand estimation and compute F

Check demand estimation (tract/zip level)
"""

import time
import numpy as np
import pandas as pd



def initial_BLP_estimation(Chain_type, capacity, heterogeneity=True, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):
    
    '''
    tract_centroids.csv : tract info
    block_data.csv : block info (needs to be sorted), ['zip', 'blkid', 'dist', 'population', 'weights', 'market_ids', 'nodes', 'logdist']
    blk_tract.csv : block-tract pairs
    agent_results_{capacity}_200_3q.csv : block-level demand estimates, based on capacity K = {8000, 10000, 12000, 15000}
    ca_blk_pharm_dist.csv : block-pharmacy
    ca_blk_{Chain_type}_dist.csv : block-chain, need to be precomputed from block_dist_chain.py (with geonear in STATA)
    '''
    
    print(f'Start initializing BLP matrices from estimation for {Chain_type} under capacity {str(capacity)}... (only need to do this once)')
    
    ### Tract-block
    tract = pd.read_csv(f'{datadir}tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int})
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv') 
    block.sort_values(by=['blkid'], inplace=True)
    blk_tract = pd.read_csv(f'{datadir}/Intermediate/blk_tract.csv', usecols=['tract', 'blkid']) 
    
    if heterogeneity:
        block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q.csv', delimiter = ",") 
    else:
        block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q_const_nodisthet.csv', delimiter = ",") 
    

    ### Distance pairs
    distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
    distdf_chain = pd.read_csv(f'{datadir}Intermediate/ca_blk_{Chain_type}_dist.csv', dtype={'locid': int, 'blkid': int}) # 

    C_total, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type=Chain_type, M=20)
    construct_F_BLP(Chain_type, capacity, heterogeneity, C_total, num_tracts, num_current_stores, num_total_stores, tract, block, blk_tract, block_utils, distdf, distdf_chain)


def demand_check(Chain_type, capacity, heterogeneity, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):

    ### Tract-block
    tract = pd.read_csv(f'{datadir}tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int})
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv') 
    block.sort_values(by=['blkid'], inplace=True)
    blk_tract = pd.read_csv(f'{datadir}/Intermediate/blk_tract.csv', usecols=['tract', 'blkid']) 

    if heterogeneity:
        block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q.csv', delimiter = ",") 
    else:
        block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q_const_nodisthet.csv', delimiter = ",")

    ### Distance pairs
    distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
    C_total, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type=Chain_type, M=20)
    
    zip_demand_checks(capacity, heterogeneity, block, block_utils, distdf)
    # tract_demand_check(capacity, heterogeneity, num_tracts, tract, block, blk_tract, block_utils, distdf)
    



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



def construct_F_BLP(Chain_type, capacity, heterogeneity, C_total, num_tracts, num_current_stores, num_total_stores, tract, block, blk_tract, block_utils, distdf, distdf_chain, M=12, resultdir='/export/storage_covidvaccine/Result/'):
    
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

            # print(i)

            tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
            blk_tract_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()
            block_id = block[block.blkid.isin(blk_tract_id)].blkid.to_list()
            block_utils_id = block_utils[block_utils.blkid.isin(blk_tract_id)].blkid.to_list()


            if len(blk_tract_id) != len(block_id) or len(block_id) != len(block_utils_id) or len(block_utils_id) != len(blk_tract_id):
                print(i, len(blk_tract_id), len(block_id), len(block_utils_id))
                common_blocks_id = set(blk_tract_id) & set(block_id) & set(block_utils_id)
            else:
                common_blocks_id = blk_tract_id


            blocks_pop = block[block.blkid.isin(common_blocks_id)].population.to_numpy()
            blocks_abd = block_utils[block_utils.blkid.isin(common_blocks_id)].abd.to_numpy()
            blocks_distcoef = block_utils[block_utils.blkid.isin(common_blocks_id)].distcoef.to_numpy()
            block_distdf = distdf[distdf.blkid.isin(common_blocks_id)]

            tract_pop = np.sum(blocks_pop)
            
            for site in M_closest_current[i]:
                
                tract_site_willingess = 0    
                dist_blocks_id = block_distdf[block_distdf.locid == site].blkid.to_list()

                ############################################################
                # Tract 361 and site 8: 85 block ids, but 87 logdists
                # e.g., 18545 is in blk_tract and distdf, but not in block and block_utils, so we need to take the subset

                # Tract 1104: has 5906 population, and two blocks (97708, 97709) from blk_tract & block_data, but none in the agent_results
                ############################################################

                if len(dist_blocks_id) != len(common_blocks_id):
                    
                    # print("****************************\n")
                    # print(i, site, len(blocks_pop), len(blocks_abd), len(dist_blocks_id))
                    # print("****************************\n")

                    # subset
                    temp_common_blocks_id = set(dist_blocks_id) & set(common_blocks_id)
                    temp_blocks_pop = block[block.blkid.isin(temp_common_blocks_id)].population.to_numpy()
                    temp_blocks_abd = block_utils[block_utils.blkid.isin(temp_common_blocks_id)].abd.to_numpy()
                    temp_blocks_distcoef = block_utils[block_utils.blkid.isin(temp_common_blocks_id)].distcoef.to_numpy()
                    temp_block_distdf = block_distdf[block_distdf.blkid.isin(temp_common_blocks_id)]
                    temp_tract_pop = np.sum(temp_blocks_pop)
                    
                    logdists = temp_block_distdf[temp_block_distdf.locid == site].logdist.to_numpy()

                    blocks_utility = temp_blocks_abd + temp_blocks_distcoef * logdists
                    tract_site_willingess = np.sum((temp_blocks_pop / temp_tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))

                else:
                    
                    logdists = block_distdf[block_distdf.locid == site].logdist.to_numpy()

                    blocks_utility = blocks_abd + blocks_distcoef * logdists
                    tract_site_willingess = np.sum((blocks_pop / tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))

                
                F_current[i][site] = tract_site_willingess

        print(f'Finished computing, time spent: {str(int(time.time()-start))}; Start exporting...\n')
        F_current_df = pd.DataFrame(F_current)
        if heterogeneity:
            F_current_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}.csv', header=False, index=False)
        else:
            F_current_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}_nodisthet.csv', header=False, index=False)

    ######################################################################


    print(f'Constructing F_BLP_{Chain_type}...\n')

    C_chains = C_total[:,num_current_stores:num_total_stores]
    M_closest_chains = np.argpartition(C_chains, M, axis=1)[:,:M]
    F_chains = np.zeros((num_tracts, num_total_stores-num_current_stores))

    start = time.time()

    for i in range(num_tracts):

        print(i)

        tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
        blk_tract_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()
        block_id = block[block.blkid.isin(blk_tract_id)].blkid.to_list()
        block_utils_id = block_utils[block_utils.blkid.isin(blk_tract_id)].blkid.to_list()


        if len(blk_tract_id) != len(block_id) or len(block_id) != len(block_utils_id) or len(block_utils_id) != len(blk_tract_id):
            print(i, len(blk_tract_id), len(block_id), len(block_utils_id))
            common_blocks_id = set(blk_tract_id) & set(block_id) & set(block_utils_id)
        else:
            common_blocks_id = blk_tract_id


        blocks_pop = block[block.blkid.isin(common_blocks_id)].population.to_numpy()
        blocks_abd = block_utils[block_utils.blkid.isin(common_blocks_id)].abd.to_numpy()
        blocks_distcoef = block_utils[block_utils.blkid.isin(common_blocks_id)].distcoef.to_numpy()
        block_distdf = distdf_chain[distdf_chain.blkid.isin(common_blocks_id)] # difference 1

        tract_pop = np.sum(blocks_pop)
            
        for site in M_closest_chains[i]:

            tract_site_willingess = 0    
            dist_blocks_id = block_distdf[block_distdf.locid == site].blkid.to_list()

            if len(dist_blocks_id) != len(common_blocks_id):
                    
                temp_common_blocks_id = set(dist_blocks_id) & set(common_blocks_id)
                temp_blocks_pop = block[block.blkid.isin(temp_common_blocks_id)].population.to_numpy()
                temp_blocks_abd = block_utils[block_utils.blkid.isin(temp_common_blocks_id)].abd.to_numpy()
                temp_blocks_distcoef = block_utils[block_utils.blkid.isin(temp_common_blocks_id)].distcoef.to_numpy()
                temp_block_distdf = block_distdf[block_distdf.blkid.isin(temp_common_blocks_id)]
                temp_tract_pop = np.sum(temp_blocks_pop)
                    
                logdists = temp_block_distdf[temp_block_distdf.locid == site].logdist.to_numpy()

                blocks_utility = temp_blocks_abd + temp_blocks_distcoef * logdists
                tract_site_willingess = np.sum((temp_blocks_pop / temp_tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))

            else:
                    
                logdists = block_distdf[block_distdf.locid == site].logdist.to_numpy()

                blocks_utility = blocks_abd + blocks_distcoef * logdists
                tract_site_willingess = np.sum((blocks_pop / tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))

            F_chains[i][site] = tract_site_willingess

    print(f'Finished computing, time spent: {str(int(time.time()-start))}; Start exporting...\n')    
    F_chains_df = pd.DataFrame(F_chains)
    if heterogeneity:
        F_chains_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}.csv', header=False, index=False)
    else:
        F_chains_df.to_csv(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}_nodisthet.csv', header=False, index=False)



def import_BLP_estimation(Chain_type, capacity, resultdir='/export/storage_covidvaccine/Result/'):

    # NOTE: WITHOUT HETEROGENEITY HASN'T UPDATED TO 5+
    # TODO THIS, RERUN INITIAL_BLP_ESTIMATION
    F_D_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}_nodisthet.csv', delimiter = ",", dtype = float) 
    F_D_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}_nodisthet.csv', delimiter = ",", dtype = float)
    F_D_total = np.concatenate((F_D_current, F_D_chain), axis = 1)

    F_DH_current = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_current_{str(capacity)}.csv', delimiter = ",", dtype = float) 
    F_DH_chain = np.genfromtxt(f'{resultdir}BLP_matrix/BLP_matrix_{Chain_type}_{str(capacity)}.csv', delimiter = ",", dtype = float)
    F_DH_total = np.concatenate((F_DH_current, F_DH_chain), axis = 1)

    return F_D_current, F_D_total, F_DH_current, F_DH_total




def tract_demand_check(capacity, heterogeneity, num_tracts, tract, block, blk_tract, block_utils, distdf, resultdir='/export/storage_covidvaccine/Result/'):
    
    print('Start tract-level demand check...\n')

    Tract_summary = []
    for i in range(num_tracts):

        tract_site_willingess = 0

        tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
        blk_tract_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()
        block_id = block[block.blkid.isin(blk_tract_id)].blkid.to_list()
        block_utils_id = block_utils[block_utils.blkid.isin(blk_tract_id)].blkid.to_list()


        if len(blk_tract_id) != len(block_id) or len(block_id) != len(block_utils_id) or len(block_utils_id) != len(blk_tract_id):
            print(i, len(blk_tract_id), len(block_id), len(block_utils_id))
            common_blocks_id = set(blk_tract_id) & set(block_id) & set(block_utils_id)
        else:
            common_blocks_id = blk_tract_id


        blocks_pop = block[block.blkid.isin(common_blocks_id)].population.to_numpy()
        blocks_abd = block_utils[block_utils.blkid.isin(common_blocks_id)].abd.to_numpy()
        blocks_distcoef = block_utils[block_utils.blkid.isin(common_blocks_id)].distcoef.to_numpy()
        tract_pop = np.sum(blocks_pop)

        # print(common_blocks_id)
        block_distdf = distdf[distdf.blkid.isin(common_blocks_id)]
        # print('****************************\n')
        # print(block_distdf)
        # print('****************************\n')
        logdists = block_distdf.groupby(['blkid'])['logdist'].min() # group by block, find min logdist
        # print(logdists)


        blocks_utility = blocks_abd + blocks_distcoef * logdists
        tract_site_willingess = np.sum((blocks_pop / tract_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))
        
        Tract_summary.append({'Tract': i, 'GEOID': tract_id, 'Num blocks': len(common_blocks_id), 'Estimated': tract_site_willingess})
        
    Tract_summary = pd.DataFrame(Tract_summary)
    if heterogeneity: Tract_summary.to_csv(f'{resultdir}/Tract_demand_check_{str(capacity)}.csv', encoding='utf-8', index=False, header=True)
    else: Tract_summary.to_csv(f'{resultdir}/Tract_demand_check_{str(capacity)}_nodisthet.csv', encoding='utf-8', index=False, header=True)
    
    




def zip_demand_checks(capacity, heterogeneity, block, block_utils, distdf, resultdir='/export/storage_covidvaccine/Result/'):
    
    print('Start zip-level demand check...\n')
    
    zips = block.zip.unique()
    Zip_summary = []
    for zip in zips:

        zip_site_willingess = 0

        block_zip_id = block[block.zip == zip].blkid.unique()
        block_id = block[block.blkid.isin(block_zip_id)].blkid.to_list()
        block_utils_id = block_utils[block_utils.blkid.isin(block_zip_id)].blkid.to_list()

        if len(block_zip_id) != len(block_id) or len(block_id) != len(block_utils_id) or len(block_utils_id) != len(block_zip_id):
            common_blocks_id = set(block_zip_id) & set(block_id) & set(block_utils_id)
        else:
            common_blocks_id = block_zip_id


        blocks_pop = block[block.blkid.isin(common_blocks_id)].population.to_numpy()
        blocks_abd = block_utils[block_utils.blkid.isin(common_blocks_id)].abd.to_numpy()
        blocks_distcoef = block_utils[block_utils.blkid.isin(common_blocks_id)].distcoef.to_numpy()
        zip_pop = np.sum(blocks_pop)

        block_distdf = distdf[distdf.blkid.isin(common_blocks_id)]   
        logdists = block_distdf.groupby(['blkid'])['logdist'].min()

        blocks_utility = blocks_abd + blocks_distcoef * logdists
        zip_site_willingess = np.sum((blocks_pop / zip_pop) * (np.exp(blocks_utility)/(1 + np.exp(blocks_utility))))

        Zip_summary.append({'Zip': zip, 'Num blocks': len(common_blocks_id), 'Estimated': zip_site_willingess})

    Zip_summary = pd.DataFrame(Zip_summary)

    if heterogeneity: Zip_summary.to_csv(f'{resultdir}/Zip_demand_check_{str(capacity)}.csv', encoding='utf-8', index=False, header=True)
    else: Zip_summary.to_csv(f'{resultdir}/Zip_demand_check_{str(capacity)}_nodisthet.csv', encoding='utf-8', index=False, header=True)



