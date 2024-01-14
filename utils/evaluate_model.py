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



def compute_distdf(Chain_type, Chain, constraint, z, R, setting_tag, path, datadir='/export/storage_covidvaccine/Data', resultdir='/export/storage_covidvaccine/Result'):
    
    print('Start computing distdf...\n')
    path = f'{path}/{constraint}'
     
    pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
    pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
    pharmacy_locations.drop(columns=['StateID'], inplace=True)

    # Import locations
    if Chain == 'Dollar':
        chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{Chain_type}.csv", usecols=['Latitude', 'Longitude', 'State'])
        chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
        
    elif Chain == 'Coffee':
        chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{Chain_type}.csv", usecols=['latitude', 'longitude', 'region'])
        chain_locations.rename(columns={'region': 'State'}, inplace=True)

    elif Chain == 'HighSchools':
        chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{Chain_type}.csv", usecols=['latitude', 'longitude', 'State'])
    else:
        print('Warning: chain name undefined\n')

    chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
    chain_locations.drop(columns=['State'], inplace=True)  

    all_locations = pd.concat([pharmacy_locations, chain_locations])
    selected_locations = all_locations[z == 1]
    selected_locations['id'] = range(selected_locations.shape[0])

    # ============================= STATA ==================================

    chainlocpath = f"{path}/ca_{Chain}_locations_total{setting_tag}.dta"
    selected_locations.to_stata(chainlocpath, write_index=False)

    baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
    chain = pd.read_stata(chainlocpath)
    chain.tail()
    len(set(chain.id))

    outpath = f"{path}/ca_blk_{Chain}_dist_total{setting_tag}.csv"
    within = 3000 # km
    limit = 50 # number of chain stores to consider
    output = subprocess.run(["stata-mp", "-b", "do", f"/mnt/phd/jihu/VaxDemandDistance/Demand/datawork/geonear_pharmacies.do", baselocpath, chainlocpath, outpath, str(within), str(limit)], capture_output=True, text=True)
    
    # ============================= STATA ==================================
    

    return 



# ===========================================================================


def construct_blocks(Chain, M, K, nsplits, flexible_consideration, flex_thresh, R, setting_tag, constraint, path, Pharmacy=False, datadir='/export/storage_covidvaccine/Data', resultdir='/export/storage_covidvaccine/Result'):

    print('Start constructing blocks...\n')
    if constraint != 'None': path = f'{path}/{constraint}'

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
        distdf = pd.read_csv(f'{path}/ca_blk_{Chain}_dist_total{setting_tag}.csv', dtype={'locid': int, 'blkid': int})


    # Block estimation
    block_utils = pd.read_csv(f'{resultdir}/Demand/agent_results{setting_tag}.csv', delimiter = ",")
    block_utils = block_utils.loc[block_utils.blkid.isin(blocks_unique), :]


    # Keep markets in both
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = de.hpi_dist_terms(df, nsplits=nsplits, add_hpi_bins=True, add_hpi_dummies=True, add_dist=False)
    df['popdensity_group'] = pd.cut(df['popdensity'], bins=[0,1000,3000,np.inf], labels=['rural', 'suburban', 'urban'], right=False)
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))


    # Flexible 
    if flexible_consideration:

        cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/cw_pop.csv")        
        distdf = distdf.merge(cw_pop[['market_ids', 'blkid']], on='blkid', how='left')
        distdf = distdf.merge(df[['market_ids', 'popdensity_group']], on='market_ids', how='left')
        distdf['popdensity_group'].fillna('rural', inplace=True)

        distdf_maxrank = distdf.groupby('blkid').head(1).reset_index(drop=True) # if flexible, max_rank = 1
        distdf_list = [distdf_maxrank]


        assert set(distdf.popdensity_group) == set(flex_thresh.keys())
        for grp in flex_thresh.keys():
            # grab distdf for this popdensity_group and within the threshold
            distdf_grp = distdf.loc[(distdf.popdensity_group == grp) & (distdf.logdist <= np.log(flex_thresh[grp])), :]
            distdf_list.append(distdf_grp)
            
        # take the union
        distdf = pd.concat(distdf_list).drop_duplicates().reset_index(drop=True)
        # sort by blkid, then logdist
        distdf.sort_values(by=['blkid', 'logdist'], inplace=True)

    else:
        distdf = distdf.groupby('blkid').head(M).reset_index(drop=True) # originally

    distdf = distdf.loc[distdf.blkid.isin(blocks_unique), :]


    # Subset blocks and add HPI
    block = block.loc[block.market_ids.isin(mkts_in_both), :]
    df = df.loc[df.market_ids.isin(mkts_in_both), :]
    distdf = distdf.loc[distdf.blkid.isin(block.blkid.unique()), :]
    block = block.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')

    return block, block_utils, distdf



# ===========================================================================


def run_assignment(Chain, M, K, nsplits, capcoef, mnl, setting_tag, constraint, block, block_utils, distdf, path, Pharmacy=False):

    print('Start assignments...\n')
    if constraint != 'None': path = f'{path}/{constraint}'

    distcoefs = block_utils.distcoef.values
    abd = block_utils.abd.values

    dist_grp = distdf.groupby('blkid')
    locs = dist_grp.locid.apply(np.array).values # list of lists of location IDs, corresponding to dists
    dists = dist_grp.logdist.apply(np.array).values # list of lists of distances, sorted ascending
    geog_pops = block.population.values
    geog_pops = np.array(geog_pops).astype(int).tolist() # updated: float to int

    test = False
    if test:
        num_rows = 3
        locs = locs[:num_rows]
        dists = dists[:num_rows]
        geog_pops = geog_pops[:num_rows]
        distcoefs = distcoefs[:num_rows]
        abd = abd[:num_rows]
        block = block[:num_rows]

    # ===========================================================================

    economy = vaxclass.Economy(locs=locs, dists=dists, geog_pops=geog_pops, max_rank=M, seed=42, mnl=mnl)
    af.random_fcfs(economy, distcoefs, abd, K, mnl, True)
    assignment = economy.assignments

    # ===========================================================================

    # NOTE: b/c of flexible_consideration the length of each array is different
    # Pad each array with zeros make the output so large


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

    if Pharmacy:
        np.savetxt(f'{path}/locs_Pharmacy{setting_tag}.csv', np.stack(locs_padded_dense, axis=0), fmt='%s')
        np.savetxt(f'{path}/dists_Pharmacy{setting_tag}.csv', np.stack(dists_padded_dense, axis=0))
        np.savetxt(f'{path}/assignment_Pharmacy{setting_tag}.csv', np.array(assignment_padded_dense), fmt='%s')
    
    else:
        np.savetxt(f'{path}/locs_{Chain}{setting_tag}.csv', np.stack(locs_padded_dense, axis=0), fmt='%s')
        np.savetxt(f'{path}/dists_{Chain}{setting_tag}.csv', np.stack(dists_padded_dense, axis=0))
        np.savetxt(f'{path}/assignment_{Chain}{setting_tag}.csv', np.array(assignment_padded_dense), fmt='%s')

    # summary_statistics(assignment, locs, dists, block, nsplits, setting_tag, path)
        
    return


# ===========================================================================


# def summary_statistics(assignment, locs, dists, block, nsplits, setting_tag, path):

#     assignment_hpi = [[assignment[j] for j in block[block.hpi_quantile == i].index] for i in range(1, nsplits+1)]
#     total_vaccination = sum(np.sum(arr) for arr in assignment)
#     total_vaccination_list = [np.sum(assignment_hpi_i) for assignment_hpi_i in assignment_hpi]
#     total_vaccination_list = np.round(np.array(total_vaccination_list) / 1000000, 4)
#     summary = {'Vaccination': total_vaccination, 
#         'Vaccination HPI1': total_vaccination_list[0], 'Vaccination HPI2': total_vaccination_list[1], 
#         'Vaccination HPI3': total_vaccination_list[2], 'Vaccination HPI4': total_vaccination_list[3]}
#     summary_df = pd.DataFrame([summary])
#     summary_df.to_csv(f'{path}/results{setting_tag}.csv', index=False)

#     return 


# ===========================================================================



def evaluate_rate(scenario, constraint, z, pc, pf, ncp, p, closest, K,
                  num_current_stores, num_total_stores, num_tracts, 
                  scale_factor, path, R = None, MIPGap = 1e-3):
    
    """
    Parameters
    ----------
    scenario : string
        "current": current stores only
        "total": current and dollar stores
        
    Demand_estimation : string
        "BLP":
        "Logit":
        "Linear":
        
    pc : array
        scaled population * distance
    
    pf : array
        scaled population * willingness
        
    ncp : array
        n copies of population vector
        
    p : array
        population vector
        
    closest : array
        0-1 vector that indicates if (i,j) is the nth closest pair
    
    K : scalar
        capacity of a single site
        
    scale_factor : scalar
        scale the value down by a factor to ensure computational feasibility
        
    path : string
        directory for results

    """


    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("Vaccination")
    m.Params.MIPGap = MIPGap
    
    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    y = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'y')
    

    ### Objective ###
    m.setObjective(quicksum(pf[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    
    
    ### Constraints ###
    if constraint == 'assigned':
        for j in range(num_stores):
            m.addConstr(quicksum(p[i] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
    elif constraint == 'vaccinated':
        for j in range(num_stores):
            m.addConstr(quicksum(pf[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j]) # TODO: double check the formula

    for i in range(num_tracts):
        m.addConstr(quicksum(y[i * num_stores + j] for j in range(num_stores)) <= 1)
          
    for k in range(num_tracts * num_stores):
        m.addConstr(y[k] <= closest[k])


    ## Solve ###
    m.update()
    start = time.time()
    m.optimize()
    end = time.time()

    ### Export ###  
    y_soln = np.zeros(num_tracts * num_stores)
    for k in range(num_tracts * num_stores):
        y_soln[k] = y[k].X    

    if R is not None:
        np.savetxt(f'{path}y_{scenario}_eval_fixR{str(R)}.csv', y_soln, delimiter=",") 
    else:
        np.savetxt(f'{path}y_{scenario}_eval.csv', y_soln, delimiter=",")


    ### Finished all ###
    m.dispose()
    

    