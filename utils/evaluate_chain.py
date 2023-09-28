#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import pandas as pd
import numpy as np
import subprocess
import geopandas as gpd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.evaluate_model import evaluate_rate
from utils.construct_F import construct_F_BLP, construct_F_LogLin
from utils.import_dist import import_dist
from utils.import_demand import import_BLP_estimation

scale_factor = 10000

try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de



def evaluate_chain_MIP(Chain_type, Model, M, K, expdirpath, constraint_list = ['assigned', 'vaccinated']):
    
    print(f'Evaluating using MIP with Chain type: {Chain_type}; Model: {Model}; M = {str(M)}, K = {str(K)}. Results stored at {expdirpath}')
    
    Population, Quartile, p_current, p_total, pc_current, pc_total, C_total, Closest_current, Closest_total, c_currentMinDist, c_totalMinDist, num_tracts, num_current_stores, num_total_stores = import_dist(Chain_type, M)
    
    F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain_type, K)
    
    f_dh_current = F_DH_current.flatten()
    f_dh_total = F_DH_total.flatten()
    pfdh_current = p_current * f_dh_current
    pfdh_total = p_total * f_dh_total

    # Import optimal z from optimziation
    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

        for opt_constr in constraint_list:

            print(f'{expdirpath}{opt_constr}/z...')
            z_total = np.genfromtxt(f'{expdirpath}{opt_constr}/z_total.csv', delimiter = ",", dtype = float)
            z_current = np.genfromtxt(f'{expdirpath}{opt_constr}/z_current.csv', delimiter = ",", dtype = float)

            eval_constr = opt_constr

            if Chain_type == 'Dollar':
                evaluate_rate(scenario = 'current', constraint = eval_constr, z = z_current,
                            pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                            closest = Closest_current, K=K, 
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores, 
                            num_tracts=num_tracts,
                            scale_factor=scale_factor,
                            path = expdirpath + opt_constr + '/')

            evaluate_rate(scenario = 'total', constraint = eval_constr, z = z_total,
                        pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                        closest = Closest_total, K=K,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath + opt_constr + '/')


    else: # MinDist

        print(f'{expdirpath}z...')
        z_total = np.genfromtxt(f'{expdirpath}z_total.csv', delimiter = ",", dtype = float)
        z_current = np.genfromtxt(f'{expdirpath}z_current.csv', delimiter = ",", dtype = float)
        
        # constraint_list = ['assigned'] # TEMP
        for eval_constr in constraint_list:

            if Chain_type == 'Dollar':
                evaluate_rate(scenario = 'current', constraint = eval_constr, z = z_current,
                            pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                            closest = Closest_current, K=K, 
                            num_current_stores=num_current_stores,
                            num_total_stores=num_total_stores, 
                            num_tracts=num_tracts,
                            scale_factor=scale_factor,
                            path = expdirpath, 
                            MIPGap = 1e-2)

            evaluate_rate(scenario = 'total', constraint = eval_constr, z = z_total,
                        pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                        closest = Closest_total, K=K,
                        num_current_stores=num_current_stores,
                        num_total_stores=num_total_stores,
                        num_tracts=num_tracts,
                        scale_factor=scale_factor,
                        path = expdirpath,
                        MIPGap = 5e-2)



    pass



def evaluate_chain_RandomFCFS(Chain_type, Model, M, K, expdirpath, constraint_list = ['assigned', 'vaccinated'], 
                            datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):

    print(f'Evaluating random order first-come-first served with Chain type: {Chain_type}; Model: {Model}; M = {str(M)}, K = {str(K)}. Results stored at {expdirpath}')
    
    # TODO: need to construct a dictionary
    temp_dict = {'Dollar': '01_DollarStores'}

    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

        for opt_constr in constraint_list:

            z_total = np.genfromtxt(f'{expdirpath}{opt_constr}/z_total.csv', delimiter = ",", dtype = float)

            # TODO: if distdf computed, no need to recompute
            compute_distdf(chain_type=temp_dict[Chain_type], chain_name=Chain_type, opt_constr=opt_constr, z=z_total, expdirpath=expdirpath)

            if Chain_type == 'Dollar' and Model == 'MaxVaxHPIDistBLP' and opt_constr == 'assigned': # only once

                block, block_utils, distdf = construct_blocks(Chain_type, M, K, opt_constr, expdirpath)
                run_assignment(Chain_type, M, K, opt_constr, block, block_utils, distdf, expdirpath)

                block, block_utils, distdf = construct_blocks(Chain_type, M, K, opt_constr, expdirpath, Pharmacy=True)
                run_assignment(Chain_type, M, K, opt_constr, block, block_utils, distdf, expdirpath, Pharmacy=True)
                
            else:

                block, block_utils, distdf = construct_blocks(Chain_type, M, K, opt_constr, expdirpath)
                run_assignment(Chain_type, M, K, opt_constr, block, block_utils, distdf, expdirpath)
            
    else: # MinDist

        z_total = np.genfromtxt(f'{expdirpath}z_total.csv', delimiter = ",", dtype = float)
        compute_distdf(chain_type=temp_dict[Chain_type], chain_name=Chain_type, opt_constr='None', z=z_total, expdirpath=expdirpath)

        block, block_utils, distdf = construct_blocks(Chain_type, M, K, 'None', expdirpath)
        run_assignment(Chain_type, M, K, 'None', block, block_utils, distdf, expdirpath)


    return



def compute_distdf(chain_type, chain_name, opt_constr, z, expdirpath, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):
    
    print('Start computing distdf...\n')

    if opt_constr == 'None': path = expdirpath
    else: path = expdirpath + opt_constr + '/'

    pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
    pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
    pharmacy_locations.drop(columns=['StateID'], inplace=True)

    chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{chain_type}.csv", usecols=['Latitude', 'Longitude', 'State'])
    chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
    chain_locations.drop(columns=['State'], inplace=True)

    all_locations = pd.concat([pharmacy_locations, chain_locations])
    selected_locations = all_locations[z == 1]
    selected_locations['id'] = range(selected_locations.shape[0])
    chainlocpath = f"{path}ca_{chain_name}_locations_total.dta"
    selected_locations.to_stata(chainlocpath, write_index=False)

    # ============================= STATA ==================================
    baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
    chainlocpath = f"{path}ca_{chain_name}_locations_total.dta"
    chain = pd.read_stata(chainlocpath)
    chain.tail()
    len(set(chain.id))

    outpath = f"{path}ca_blk_{chain_name}_dist_total.csv"
    within = 2000 # km
    limit = 30 # number of chain stores to consider
    output = subprocess.run(["stata-mp", "-b", "do", f"/mnt/phd/jihu/VaxDemandDistance/Demand/datawork/geonear_pharmacies.do", baselocpath, chainlocpath, outpath, str(within), str(limit)], capture_output=True, text=True)
    # ============================= STATA ==================================
    

    return 



def construct_blocks(Chain_type, M, K, opt_constr, expdirpath, num_current_stores = 4035, nsplits = 3, Pharmacy=False, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):

    print('Start constructing blocks...\n')

    if opt_constr == 'None': path = expdirpath
    else: path = expdirpath + opt_constr + '/'

    # Block basic info
    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)

    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)


    # Distance pairs
    if Pharmacy:
        distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
    else:
        distdf = pd.read_csv(f'{path}ca_blk_{Chain_type}_dist_total.csv', dtype={'locid': int, 'blkid': int})

        # this should be independent of model
        # distdf = pd.read_csv(f'{resultdir}MaxVaxHPIDistBLP/M{M}_K{K}/{Chain_type}/{opt_constr}/ca_blk_{Chain_type}_dist_total.csv', dtype={'locid': int, 'blkid': int})
        
        # distdf_pharmacy = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
        # distdf_chain = pd.read_csv(f'{datadir}/Intermediate/ca_blk_{Chain_type}_dist.csv', dtype={'locid': int, 'blkid': int})
        # distdf_chain.locid = distdf_chain.locid + num_current_stores
        # distdf = pd.concat([distdf_pharmacy, distdf_chain])

    distdf = distdf.groupby('blkid').head(M).reset_index(drop=True)
    distdf = distdf.loc[distdf.blkid.isin(blocks_unique), :]


    # Block estimation
    block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{K}_200_3q.csv', delimiter = ",") 
    block_utils = block_utils.loc[block_utils.blkid.isin(blocks_unique), :]


    # Keep markets in both
    df = pd.read_csv(f"{datadir}Analysis/Demand/demest_data.csv")
    df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=False, add_dist=False)
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))
    # print("Number of markets:", len(mkts_in_both))


    # Subset blocks and add HPI
    block = block.loc[block.market_ids.isin(mkts_in_both), :]
    df = df.loc[df.market_ids.isin(mkts_in_both), :]
    distdf = distdf.loc[distdf.blkid.isin(block.blkid.unique()), :]
    block = block.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')

    return block, block_utils, distdf



def run_assignment(Chain_type, M, K, opt_constr, block, block_utils, distdf, expdirpath, Pharmacy=False):

    print('Start assignments...\n')

    if opt_constr == 'None': path = expdirpath
    else: path = expdirpath + opt_constr + '/'

    distcoefs = block_utils.distcoef.values
    abd = block_utils.abd.values

    dist_grp = distdf.groupby('blkid')
    locs = dist_grp.locid.apply(np.array).values # list of lists of location IDs, corresponding to dists
    dists = dist_grp.logdist.apply(np.array).values # list of lists of distances, sorted ascending
    geog_pops = block.population.values
    geog_pops = np.array(geog_pops).astype(int).tolist() # updated: float to int

    economy = vaxclass.Economy(locs, dists, geog_pops, M)
    af.random_fcfs_eval(economy, distcoefs, abd, K)
    af.assignment_stats_eval(economy, M)

    if Pharmacy:
        np.savetxt(f'{expdirpath}locs_{K}_Pharmacy.csv', np.stack(locs, axis=0), fmt='%s')
        np.savetxt(f'{expdirpath}dists_{K}_Pharmacy.csv', np.stack(dists, axis=0))
        np.savetxt(f'{expdirpath}assignment_{K}_Pharmacy.csv', np.array(economy.assignments), fmt='%s')
    
    else:
        np.savetxt(f'{path}locs_{K}_{Chain_type}.csv', np.stack(locs, axis=0), fmt='%s')
        np.savetxt(f'{path}dists_{K}_{Chain_type}.csv', np.stack(dists, axis=0))
        np.savetxt(f'{path}assignment_{K}_{Chain_type}.csv', np.array(economy.assignments), fmt='%s')
        
