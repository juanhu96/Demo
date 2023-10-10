#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu
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



def import_dataset(nsplits, datadir):

    '''
    Import block, tract related dataset
    '''

    print('Importing dataset...\n')

    block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
    blocks_unique = np.unique(block.blkid.values)
    markets_unique = np.unique(block.market_ids.values)
    block = block.loc[block.blkid.isin(blocks_unique), :]
    block.sort_values(by=['blkid'], inplace=True)
    
    ### Keep markets in both
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=False, add_dist=False)
    df_temp = df.copy()
    df = df.loc[df.market_ids.isin(markets_unique), :]
    mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))

    ### Subset blocks and add HPI
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
    tract_hpi['population'] = tract_hpi['population'].astype(int)

    return df, df_temp, block, tract_hpi



# ====================================================================================



def import_solution(path, Chain_type, K, num_tracts, num_total_stores, num_current_stores, eval_constr='vaccinated', R=None, Pharmacy=False):
    
    '''
    z, y: results from first/second stage
    dists, assignment: results from second stage
    '''
    
    print('Importing solution...\n')

    if not Pharmacy:

        if R is not None:
            locs = np.genfromtxt(f'{path}locs_{K}_{Chain_type}_fixR{str(R)}.csv', delimiter = "")
            dists = np.genfromtxt(f'{path}dists_{K}_{Chain_type}_fixR{str(R)}.csv', delimiter = "")
            assignment = np.genfromtxt(f'{path}assignment_{K}_{Chain_type}_fixR{str(R)}.csv', delimiter = "")

            z = np.genfromtxt(f'{path}z_total_fixR{str(R)}.csv', delimiter = ",", dtype = float)
            y = np.genfromtxt(f'{path}y_total_fixR{str(R)}.csv', delimiter = ",", dtype = float)
            # NOTE: we did not perform second-stage evalution on R scenarios yet
            # y_eval = np.genfromtxt(f'{path}y_total_eval_{eval_constr}_fixR.csv', delimiter = ",", dtype = float)
            mat_y = np.reshape(y, (num_tracts, num_total_stores))
            # mat_y_eval = np.reshape(y_eval, (num_tracts, num_total_stores))

            return z, mat_y, locs, dists, assignment 
        
        else:
            locs = np.genfromtxt(f'{path}locs_{K}_{Chain_type}.csv', delimiter = "")
            dists = np.genfromtxt(f'{path}dists_{K}_{Chain_type}.csv', delimiter = "")
            assignment = np.genfromtxt(f'{path}assignment_{K}_{Chain_type}.csv', delimiter = "")

            z = np.genfromtxt(f'{path}z_total.csv', delimiter = ",", dtype = float)
            y = np.genfromtxt(f'{path}y_total.csv', delimiter = ",", dtype = float)
            y_eval = np.genfromtxt(f'{path}y_total_eval_{eval_constr}.csv', delimiter = ",", dtype = float)
            mat_y = np.reshape(y, (num_tracts, num_total_stores))
            mat_y_eval = np.reshape(y_eval, (num_tracts, num_total_stores))

            return z, mat_y, mat_y_eval, locs, dists, assignment 

    else:
        locs = np.genfromtxt(f'{path}locs_{K}_Pharmacy.csv', delimiter = "")
        dists = np.genfromtxt(f'{path}dists_{K}_Pharmacy.csv', delimiter = "")
        assignment = np.genfromtxt(f'{path}assignment_{K}_Pharmacy.csv', delimiter = "")

        z = np.genfromtxt(f'{path}{eval_constr}/z_current.csv', delimiter = ",", dtype = float)
        y = np.genfromtxt(f'{path}{eval_constr}/y_current.csv', delimiter = ",", dtype = float)
        y_eval = np.genfromtxt(f'{path}{eval_constr}/y_current_eval_{eval_constr}.csv', delimiter = ",", dtype = float)
        mat_y = np.reshape(y, (num_tracts, num_current_stores))
        mat_y_eval = np.reshape(y_eval, (num_tracts, num_current_stores))       
            
        return z, mat_y, mat_y_eval, locs, dists, assignment



# ====================================================================================



def import_locations(df, Chain_type, Chain_name_list={'Dollar': '01_DollarStores', 'Coffee': '04_Coffee', 'HighSchools': '09_HighSchools'}, datadir='/export/storage_covidvaccine/Data/'):

    '''
    Import pharmacies/chain locations and distance matrices
    '''

    df.rename(columns = {"zip": "zip_code"}, inplace = True)
    df['zip_code'] = df['zip_code'].astype("string")
    
    pharmacy_locations = pd.read_csv(f"{datadir}Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'zip_code', 'StateID'])
    pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
    pharmacy_locations.drop(columns=['StateID'], inplace=True)
    pharmacy_locations['zip_code'] = pharmacy_locations['zip_code'].astype("string")
    pharmacy_locations = pharmacy_locations.merge(df[['zip_code', 'hpi_quantile']], on='zip_code', how='left')

    Chain_name = Chain_name_list[Chain_type]

    ## NOTE: TEMP 
    if Chain_type == 'Dollar':
        chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{Chain_name}.csv", usecols=['Latitude', 'Longitude', 'Zip_Code', 'State'])
        chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude', 'Zip_Code': 'zip_code'}, inplace=True)
        
    elif Chain_type == 'Coffee':
        chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{Chain_name}.csv", usecols=['latitude', 'longitude', 'postal_code', 'region'])
        chain_locations.rename(columns={'postal_code': 'zip_code', 'region': 'State'}, inplace=True)

    elif Chain_type == 'HighSchools':
        chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{Chain_name}.csv", usecols=['latitude', 'longitude', 'Zip', 'State'])
        chain_locations.rename(columns={'Zip': 'zip_code'}, inplace=True)
    else:
        print('Warning: chain name undefined\n')

    chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
    chain_locations.drop(columns=['State'], inplace=True)
    chain_locations['zip_code'] = chain_locations['zip_code'].astype("string")
    chain_locations = chain_locations.merge(df[['zip_code', 'hpi_quantile']], on='zip_code', how='left')

    # ====================================================================================

    C_current = np.genfromtxt(f'{datadir}CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current = C_current.astype(int)
    C_current = C_current.T
    num_tracts, num_current_stores = np.shape(C_current)
    
    C_chains = np.genfromtxt(f'{datadir}CA_dist_matrix_{Chain_type}.csv', delimiter = ",", dtype = float)
    C_chains = C_chains.astype(int)
    C_chains = C_chains.T
    num_tracts, num_chain_stores = np.shape(C_chains)
    C_chains = np.where(C_chains < 0, 1317574, C_chains) # High schools
    C_total = np.concatenate((C_current, C_chains), axis = 1)
    num_total_stores = num_current_stores + num_chain_stores
            
    C_current_walkable = np.where(C_current < 1600, 1, 0)
    C_chains_walkable = np.where(C_chains < 1600, 1, 0)
    C_total_walkable = np.where(C_total < 1600, 1, 0)

    return pharmacy_locations, chain_locations, num_tracts, num_current_stores, num_total_stores, C_current, C_total, C_current_walkable, C_total_walkable



# ====================================================================================



def create_row_randomFCFS(Scenario, Model, Chain_type, M, K, opt_constr, stage, z, block, locs, dists, assignment, pharmacy_locations, chain_locations, num_current_stores, num_total_stores):

    # stores breakdown
    if opt_constr != 'none':
        R = num_current_stores - sum(z[0 : num_current_stores])
        closed_pharmacy = pharmacy_locations[z[0:num_current_stores] != 1]
        selected_chains = chain_locations[z[num_current_stores:] == 1]
        R_chains = selected_chains.shape[0]


        pharmacy_1 = len(closed_pharmacy[closed_pharmacy.hpi_quantile == 1].index)
        pharmacy_2 = len(closed_pharmacy[closed_pharmacy.hpi_quantile == 2].index)
        pharmacy_3 = len(closed_pharmacy[closed_pharmacy.hpi_quantile == 3].index)

        chains_1 = selected_chains[selected_chains.hpi_quantile == 1].shape[0]
        chains_2 = selected_chains[selected_chains.hpi_quantile == 2].shape[0]
        chains_3 = selected_chains[selected_chains.hpi_quantile == 3].shape[0]

    else:
        R, R_chains = 0, 0
        pharmacy_1, pharmacy_2, pharmacy_3 = 0, 0, 0
        chains_1, chains_2, chains_3 = 0, 0, 0
    

    population = sum(block.population)
    population_1 = sum(block[block.hpi_quantile == 1].population)
    population_2 = sum(block[block.hpi_quantile == 2].population)
    population_3 = sum(block[block.hpi_quantile == 3].population)
    total_population = np.round(np.array([population, population_1, population_2, population_3]) / 1000000, 2)

    dists_1 = dists[block.hpi_quantile == 1]
    dists_2 = dists[block.hpi_quantile == 2]
    dists_3 = dists[block.hpi_quantile == 3]
    assignment_1 = assignment[block.hpi_quantile == 1]
    assignment_2 = assignment[block.hpi_quantile == 2]
    assignment_3 = assignment[block.hpi_quantile == 3]

    locs_pharmacy = np.where(locs <= num_current_stores - R_chains, 1, 0) 
    locs_chain = np.where(locs > num_current_stores - R_chains, 1, 0) # the R greatest ones are dollar stores
    assignment_pharmacy = np.multiply(assignment, locs_pharmacy)
    assignment_chain  = np.multiply(assignment, locs_chain)

    assignment_pharmacy_1 = assignment_pharmacy[block.hpi_quantile == 1]
    assignment_pharmacy_2 = assignment_pharmacy[block.hpi_quantile == 2]
    assignment_pharmacy_3 = assignment_pharmacy[block.hpi_quantile == 3]

    assignment_chain_1 = assignment_chain[block.hpi_quantile == 1]
    assignment_chain_2 = assignment_chain[block.hpi_quantile == 2]
    assignment_chain_3 = assignment_chain[block.hpi_quantile == 3]

    ## Total vaccination & rates
    vaccination = np.sum(assignment)
    vaccination_1 = np.sum(assignment_1)
    vaccination_2 = np.sum(assignment_2)
    vaccination_3 = np.sum(assignment_3)
    total_vaccination = np.round(np.array([vaccination, vaccination_1, vaccination_2, vaccination_3]) / 1000000, 4)

    rate = vaccination / population
    rate_1 = vaccination_1 / population_1
    rate_2 = vaccination_2 / population_3
    rate_3 = vaccination_3 / population_3
    total_rate = np.round(np.array([rate, rate_1, rate_2, rate_3]) * 100, 3)


    ## Pharmacy vaccination & rates
    vaccination_pharmacy = np.sum(assignment_pharmacy)
    vaccination_pharmacy_1 = np.sum(assignment_pharmacy_1)
    vaccination_pharmacy_2 = np.sum(assignment_pharmacy_2)
    vaccination_pharmacy_3 = np.sum(assignment_pharmacy_3)
    total_vaccination_pharmacy = np.round(np.array([vaccination_pharmacy, vaccination_pharmacy_1, vaccination_pharmacy_2, vaccination_pharmacy_3]) / 1000000, 2)

    rate_pharmacy = vaccination_pharmacy / population
    rate_pharmacy_1 = vaccination_pharmacy_1 / population_1
    rate_pharmacy_2 = vaccination_pharmacy_2 / population_3
    rate_pharmacy_3 = vaccination_pharmacy_3 / population_3
    total_pharmacy_rate = np.round(np.array([rate_pharmacy, rate_pharmacy_1, rate_pharmacy_2, rate_pharmacy_3]) * 100, 2)
    

    ## Chain vaccination & rates
    vaccination_chain = np.sum(assignment_chain)
    vaccination_chain_1 = np.sum(assignment_chain_1)
    vaccination_chain_2 = np.sum(assignment_chain_2)
    vaccination_chain_3 = np.sum(assignment_chain_3)
    total_vaccination_chain = np.round(np.array([vaccination_chain, vaccination_chain_1, vaccination_chain_2, vaccination_chain_3]) / 1000000, 2)

    rate_chain = vaccination_chain / population
    rate_chain_1 = vaccination_chain_1 / population_1
    rate_chain_2 = vaccination_chain_2 / population_3
    rate_chain_3 = vaccination_chain_3 / population_3
    total_chain_rate = np.round(np.array([rate_chain, rate_chain_1, rate_chain_2, rate_chain_3]) * 100, 2)


    # walkable (< 1.6km)
    dists_walkable = np.where(np.exp(dists) < 1.6, 1, 0)
    assignment_walkable = assignment * dists_walkable
    assignment_walkable_1 = assignment_walkable[block.hpi_quantile == 1]
    assignment_walkable_2 = assignment_walkable[block.hpi_quantile == 2]
    assignment_walkable_3 = assignment_walkable[block.hpi_quantile == 3]

    vaccination_walkable = np.sum(assignment_walkable)
    vaccination_walkable_1 = np.sum(assignment_walkable_1)
    vaccination_walkable_2 = np.sum(assignment_walkable_2)
    vaccination_walkable_3 = np.sum(assignment_walkable_3)
    total_vaccination_walkable = np.round(np.array([vaccination_walkable, vaccination_walkable_1, vaccination_walkable_2, vaccination_walkable_3]) / 1000000, 4)

    rate_walkable = vaccination_walkable / population
    rate_walkable_1 = vaccination_walkable_1 / population_1
    rate_walkable_2 = vaccination_walkable_2 / population_3
    rate_walkable_3 = vaccination_walkable_3 / population_3
    total_rate_walkable = np.round(np.array([rate_walkable, rate_walkable_1, rate_walkable_2, rate_walkable_3]) * 100, 3)

    # walkable (pharmacy)
    dists_walkable = np.where(np.exp(dists) < 1.6, 1, 0)
    assignment_pharmacy_walkable = assignment_pharmacy * dists_walkable
    assignment_pharmacy_walkable_1 = assignment_pharmacy_walkable[block.hpi_quantile == 1]
    assignment_pharmacy_walkable_2 = assignment_pharmacy_walkable[block.hpi_quantile == 2]
    assignment_pharmacy_walkable_3 = assignment_pharmacy_walkable[block.hpi_quantile == 3]

    vaccination_pharmacy_walkable = np.sum(assignment_pharmacy_walkable)
    vaccination_pharmacy_walkable_1 = np.sum(assignment_pharmacy_walkable_1)
    vaccination_pharmacy_walkable_2 = np.sum(assignment_pharmacy_walkable_2)
    vaccination_pharmacy_walkable_3 = np.sum(assignment_pharmacy_walkable_3)
    total_vaccination_pharmacy_walkable = np.round(np.array([vaccination_pharmacy_walkable, vaccination_pharmacy_walkable_1, vaccination_pharmacy_walkable_2, vaccination_pharmacy_walkable_3]) / 1000000, 2)

    # walkable (chain)
    dists_walkable = np.where(np.exp(dists) < 1.6, 1, 0)
    assignment_chain_walkable = assignment_chain * dists_walkable
    assignment_chain_walkable_1 = assignment_chain_walkable[block.hpi_quantile == 1]
    assignment_chain_walkable_2 = assignment_chain_walkable[block.hpi_quantile == 2]
    assignment_chain_walkable_3 = assignment_chain_walkable[block.hpi_quantile == 3]

    vaccination_chain_walkable = np.sum(assignment_chain_walkable)
    vaccination_chain_walkable_1 = np.sum(assignment_chain_walkable_1)
    vaccination_chain_walkable_2 = np.sum(assignment_chain_walkable_2)
    vaccination_chain_walkable_3 = np.sum(assignment_chain_walkable_3)
    total_vaccination_chain_walkable = np.round(np.array([vaccination_chain_walkable, vaccination_chain_walkable_1, vaccination_chain_walkable_2, vaccination_chain_walkable_3]) / 1000000, 2)

    # dists is the log dist (km)
    avg_dist = np.sum(assignment * np.exp(dists)) / vaccination
    avg_dist_1 = np.sum(assignment_1 * np.exp(dists_1)) / vaccination_1
    avg_dist_2 = np.sum(assignment_2 * np.exp(dists_2)) / vaccination_2
    avg_dist_3 = np.sum(assignment_3 * np.exp(dists_3)) / vaccination_3
    total_avg_dist = np.round(np.array([avg_dist, avg_dist_1, avg_dist_2, avg_dist_3]), 2)



    chain_summary = {'Model': Model, 'Chain': Scenario,
                     'Opt Constr': opt_constr, 'Stage': stage,
                     'M': M, 'K': K, 'Pharmacies replaced': R,
                     'Pharmacies replaced HPI 1': pharmacy_1, 'Pharmacies replaced HPI 2': pharmacy_2, 
                     'Pharmacies replaced HPI 3': pharmacy_3,
                     'Stores opened HPI 1': chains_1, 'Stores opened HPI 2': chains_2, 
                     'Stores opened HPI 3': chains_3,
                     'Vaccination': total_vaccination[0], 
                     'Vaccination HPI1': total_vaccination[1], 'Vaccination HPI2': total_vaccination[2], 
                     'Vaccination HPI3': total_vaccination[3],
                     'Rate': total_rate[0],
                     'Rate HPI1': total_rate[1], 'Rate HPI2': total_rate[2], 
                     'Rate HPI3': total_rate[3],
                     'Vaccination Walkable': total_vaccination_walkable[0], 
                     'Vaccination Walkable HPI1': total_vaccination_walkable[1], 'Vaccination Walkable HPI2': total_vaccination_walkable[2], 
                     'Vaccination Walkable HPI3': total_vaccination_walkable[3],
                     'Vaccination Walkable rate HPI1': total_rate_walkable[1], 'Vaccination Walkable rate HPI2': total_rate_walkable[2],
                     'Vaccination Walkable rate HPI3': total_rate_walkable[3],
                     'Pharmacy Vaccination': total_vaccination_pharmacy[0], 
                     'Pharmacy Vaccination HPI1': total_vaccination_pharmacy[1], 'Pharmacy Vaccination HPI2': total_vaccination_pharmacy[2], 
                     'Pharmacy Vaccination HPI3': total_vaccination_pharmacy[3],
                     'Pharmacy Rate': total_pharmacy_rate[0],
                     'Pharmacy Rate HPI1': total_pharmacy_rate[1], 'Pharmacy Rate HPI2': total_pharmacy_rate[2], 
                     'Pharmacy Rate HPI3': total_pharmacy_rate[3],
                     'Chain Vaccination': total_vaccination_chain[0], 
                     'Chain Vaccination HPI1': total_vaccination_chain[1], 'Chain Vaccination HPI2': total_vaccination_chain[2], 
                     'Chain Vaccination HPI3': total_vaccination_chain[3],
                     'Chain Rate': total_chain_rate[0],
                     'Chain Rate HPI1': total_chain_rate[1], 'Chain Rate HPI2': total_chain_rate[2], 
                     'Chain Rate HPI3': total_chain_rate[3],
                     'Pharmacy Vaccination Walkable': total_vaccination_pharmacy_walkable[0], 
                     'Pharmacy Vaccination Walkable HPI1': total_vaccination_pharmacy_walkable[1], 'Pharmacy Vaccination Walkable HPI2': total_vaccination_pharmacy_walkable[2], 
                     'Pharmacy Vaccination Walkable HPI3': total_vaccination_pharmacy_walkable[3],
                     'Chain Vaccination Walkable': total_vaccination_chain_walkable[0], 
                     'Chain Vaccination Walkable HPI1': total_vaccination_chain_walkable[1], 'Chain Vaccination Walkable HPI2': total_vaccination_chain_walkable[2], 
                     'Chain Vaccination Walkable HPI3': total_vaccination_chain_walkable[3],
                     'Average distance': total_avg_dist[0],
                     'Average distance HPI1': total_avg_dist[1], 'Average distance HPI2': total_avg_dist[2],
                     'Average distance HPI3': total_avg_dist[3]}
    
    return chain_summary



# ====================================================================================



def create_row_MIP(Scenario, Model, Chain_type, M, K, opt_constr, stage, CA_TRACT, mat_y_hpi, z, F_DH, C, C_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores):

    '''
    Computing summary statistics for results from optimization, which are tract-level results
    '''

    ## Population
    total_population = sum(CA_TRACT['population'])
    population1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['population'].values)
    population2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['population'].values)
    population3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['population'].values)
    population_vec = [total_population, population1, population2, population3]


    ## Total vaccination
    total_rate_hpi = np.sum(np.multiply(F_DH, mat_y_hpi), axis = 1)
    CA_TRACT['Rate_HPI'] = total_rate_hpi
    CA_TRACT['Vaccinated_Population_HPI'] = CA_TRACT['Rate_HPI'] * CA_TRACT['population']
    
    total_rate_hpi = sum(CA_TRACT['Vaccinated_Population_HPI'].values) / total_population
    total_rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccinated_Population_HPI'].values) / population1
    total_rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccinated_Population_HPI'].values) / population2
    total_rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccinated_Population_HPI'].values) / population3

    total_rate = np.array([[total_rate_hpi, total_rate_hpi1, total_rate_hpi2, total_rate_hpi3]])
    total_vaccination = np.round(total_rate * population_vec / 1000000,2)[0]
    total_rate = np.round(total_rate * 100)[0]


    ## Total vaccination by pharmacy
    pharmacy_rate_hpi = np.sum(np.multiply(F_DH[:, 0:num_current_stores], mat_y_hpi[:, 0:num_current_stores]), axis = 1)
    CA_TRACT['Pharmacy_Rate_HPI'] = pharmacy_rate_hpi
    CA_TRACT['Pharmacy_Vaccinated_Population_HPI'] = CA_TRACT['Pharmacy_Rate_HPI'] * CA_TRACT['population']

    pharmacy_rate_hpi = sum(CA_TRACT['Pharmacy_Vaccinated_Population_HPI'].values) / total_population
    pharmacy_rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Pharmacy_Vaccinated_Population_HPI'].values) / population1
    pharmacy_rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Pharmacy_Vaccinated_Population_HPI'].values) / population2
    pharmacy_rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Pharmacy_Vaccinated_Population_HPI'].values) / population3

    pharmacy_rate = np.array([[pharmacy_rate_hpi, pharmacy_rate_hpi1, pharmacy_rate_hpi2, pharmacy_rate_hpi3]])
    pharmacy_vaccination = np.round(pharmacy_rate * population_vec / 1000000,2)[0]
    pharmacy_rate = np.round(pharmacy_rate * 100)[0]
    

    ## Total vaccination by chain
    chain_rate_hpi = np.sum(np.multiply(F_DH[:, num_current_stores:num_total_stores], mat_y_hpi[:, num_current_stores:num_total_stores]), axis = 1)
    CA_TRACT['Chain_Rate_HPI'] = chain_rate_hpi
    CA_TRACT['Chain_Vaccinated_Population_HPI'] = CA_TRACT['Chain_Rate_HPI'] * CA_TRACT['population']

    chain_rate_hpi = sum(CA_TRACT['Chain_Vaccinated_Population_HPI'].values) / total_population
    chain_rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Chain_Vaccinated_Population_HPI'].values) / population1
    chain_rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Chain_Vaccinated_Population_HPI'].values) / population2
    chain_rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Chain_Vaccinated_Population_HPI'].values) / population3

    chain_rate = np.array([[chain_rate_hpi, chain_rate_hpi1, chain_rate_hpi2, chain_rate_hpi3]])
    chain_vaccination = np.round(chain_rate * population_vec / 1000000,2)[0]
    chain_rate = np.round(chain_rate * 100)[0]


    ## Walkable vaccination
    rate_walkable_hpi = np.sum(np.multiply(C_walkable, np.multiply(F_DH, mat_y_hpi)), axis =1) 
    CA_TRACT['Vaccinate_Walkable_HPI'] = rate_walkable_hpi
    CA_TRACT['Vaccination_Walkable_HPI'] = CA_TRACT['Vaccinate_Walkable_HPI'] * CA_TRACT['population']    

    rate_walkable_hpi = sum(CA_TRACT['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccination_Walkable_HPI'].values)
    
    rate_walkable_list = np.array([[rate_walkable_hpi, rate_walkable_hpi1, rate_walkable_hpi2, rate_walkable_hpi3]])
    total_vaccination_walkable = np.round(rate_walkable_list / 1000000, 2)[0]
    total_rate_walkable = np.round(rate_walkable_list[0] / population_vec * 100)


    ## Walkable vaccination by pharmacy
    pharmacy_rate_walkable_hpi = np.sum(np.multiply(C_walkable[:, 0:num_current_stores], np.multiply(F_DH[:, 0:num_current_stores], mat_y_hpi[:, 0:num_current_stores])), axis =1) 
    CA_TRACT['Pharmacy_Vaccinate_Walkable_HPI'] = pharmacy_rate_walkable_hpi
    CA_TRACT['Pharmacy_Vaccination_Walkable_HPI'] = CA_TRACT['Pharmacy_Vaccinate_Walkable_HPI'] * CA_TRACT['population']    

    pharmacy_rate_walkable_hpi = sum(CA_TRACT['Pharmacy_Vaccination_Walkable_HPI'].values)
    pharmacy_rate_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Pharmacy_Vaccination_Walkable_HPI'].values)
    pharmacy_rate_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Pharmacy_Vaccination_Walkable_HPI'].values)
    pharmacy_rate_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Pharmacy_Vaccination_Walkable_HPI'].values)
    
    pharmacy_rate_walkable_list = np.array([[pharmacy_rate_walkable_hpi, pharmacy_rate_walkable_hpi1, pharmacy_rate_walkable_hpi2, pharmacy_rate_walkable_hpi3]])
    pharmacy_vaccination_walkable = np.round(pharmacy_rate_walkable_list / 1000000, 2)[0]
    pharmacy_rate_walkable = np.round(pharmacy_rate_walkable_list[0] / population_vec * 100)


    ## Walkable vaccination by chain
    chain_rate_walkable_hpi = np.sum(np.multiply(C_walkable[:, num_current_stores:num_total_stores], np.multiply(F_DH[:, num_current_stores:num_total_stores], mat_y_hpi[:, num_current_stores:num_total_stores])), axis =1) 
    CA_TRACT['Chain_Vaccinate_Walkable_HPI'] = chain_rate_walkable_hpi
    CA_TRACT['Chain_Vaccination_Walkable_HPI'] = CA_TRACT['Chain_Vaccinate_Walkable_HPI'] * CA_TRACT['population']    

    chain_rate_walkable_hpi = sum(CA_TRACT['Chain_Vaccination_Walkable_HPI'].values)
    chain_rate_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Chain_Vaccination_Walkable_HPI'].values)
    chain_rate_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Chain_Vaccination_Walkable_HPI'].values)
    chain_rate_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Chain_Vaccination_Walkable_HPI'].values)
    
    chain_rate_walkable_list = np.array([[chain_rate_walkable_hpi, chain_rate_walkable_hpi1, chain_rate_walkable_hpi2, chain_rate_walkable_hpi3]])
    chain_vaccination_walkable = np.round(chain_rate_walkable_list / 1000000, 2)[0]
    chain_rate_walkable = np.round(chain_rate_walkable_list[0] / population_vec * 100)


    ## Distance among vaccinated
    # assigned_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi), axis = 1) / np.sum(mat_y_hpi, axis = 1), posinf=0)
    assigned_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, np.multiply(F_DH, mat_y_hpi)), axis = 1) / np.sum(np.multiply(F_DH, mat_y_hpi), axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Assigned'] = assigned_dist_hpi
    CA_TRACT['Assigned_Population_HPI'] = CA_TRACT['population'] * np.sum(mat_y_hpi, axis = 1)
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Assigned'] * CA_TRACT['Assigned_Population_HPI']
    
    assigned_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Assigned_Population_HPI'].values)
    assigned_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_HPI'])
    assigned_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_HPI'])
    assigned_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_HPI'])
    
    assigned_dist_actual = np.array([[assigned_dist_hpi, assigned_dist_hpi1, assigned_dist_hpi2, assigned_dist_hpi3]])
    total_avg_dist = np.round(assigned_dist_actual/1000, 2)[0]


    ## stores breakdown
    if opt_constr != 'none':
        R = num_current_stores - sum(z[0 : num_current_stores])
        closed_pharmacy = pharmacy_locations[z[0:num_current_stores] != 1]
        selected_chains = chain_locations[z[num_current_stores:] == 1]

        pharmacy_1 = len(closed_pharmacy[closed_pharmacy.hpi_quantile == 1].index)
        pharmacy_2 = len(closed_pharmacy[closed_pharmacy.hpi_quantile == 2].index)
        pharmacy_3 = len(closed_pharmacy[closed_pharmacy.hpi_quantile == 3].index)

        chains_1 = selected_chains[selected_chains.hpi_quantile == 1].shape[0]
        chains_2 = selected_chains[selected_chains.hpi_quantile == 2].shape[0]
        chains_3 = selected_chains[selected_chains.hpi_quantile == 3].shape[0]

    else:
        R = 0
        pharmacy_1, pharmacy_2, pharmacy_3 = 0, 0, 0
        chains_1, chains_2, chains_3 = 0, 0, 0


    chain_summary = {'Model': Model, 'Chain': Scenario,
                     'Opt Constr': opt_constr, 'Stage': stage,
                     'M': M, 'K': K, 'Pharmacies replaced': R,
                     'Pharmacies replaced HPI 1': pharmacy_1, 'Pharmacies replaced HPI 2': pharmacy_2, 
                     'Pharmacies replaced HPI 3': pharmacy_3,
                     'Stores opened HPI 1': chains_1, 'Stores opened HPI 2': chains_2, 
                     'Stores opened HPI 3': chains_3,
                     'Vaccination': total_vaccination[0], 
                     'Vaccination HPI1': total_vaccination[1], 'Vaccination HPI2': total_vaccination[2], 
                     'Vaccination HPI3': total_vaccination[3],
                     'Rate': total_rate[0],
                     'Rate HPI1': total_rate[1], 'Rate HPI2': total_rate[2], 
                     'Rate HPI3': total_rate[3],
                     'Vaccination Walkable': total_vaccination_walkable[0], 
                     'Vaccination Walkable HPI1': total_vaccination_walkable[1], 'Vaccination Walkable HPI2': total_vaccination_walkable[2], 
                     'Vaccination Walkable HPI3': total_vaccination_walkable[3],
                     'Vaccination Walkable rate HPI1': total_rate_walkable[1], 'Vaccination Walkable rate HPI2': total_rate_walkable[2],
                     'Vaccination Walkable rate HPI3': total_rate_walkable[3],
                     'Pharmacy Vaccination': pharmacy_vaccination[0], 
                     'Pharmacy Vaccination HPI1': pharmacy_vaccination[1], 'Pharmacy Vaccination HPI2': pharmacy_vaccination[2], 
                     'Pharmacy Vaccination HPI3': pharmacy_vaccination[3],
                     'Pharmacy Rate': pharmacy_rate[0],
                     'Pharmacy Rate HPI1': pharmacy_rate[1], 'Pharmacy Rate HPI2': pharmacy_rate[2], 
                     'Pharmacy Rate HPI3': pharmacy_rate[3],
                     'Chain Vaccination': chain_vaccination[0], 
                     'Chain Vaccination HPI1': chain_vaccination[1], 'Chain Vaccination HPI2': chain_vaccination[2], 
                     'Chain Vaccination HPI3': chain_vaccination[3],
                     'Chain Rate': chain_rate[0],
                     'Chain Rate HPI1': chain_rate[1], 'Chain Rate HPI2': chain_rate[2], 
                     'Chain Rate HPI3': chain_rate[3],
                     'Pharmacy Vaccination Walkable': pharmacy_vaccination_walkable[0], 
                     'Pharmacy Vaccination Walkable HPI1': pharmacy_vaccination_walkable[1], 'Pharmacy Vaccination Walkable HPI2': pharmacy_vaccination_walkable[2], 
                     'Pharmacy Vaccination Walkable HPI3': pharmacy_vaccination_walkable[3],
                    #  'Pharmacy Vaccination Walkable rate HPI1': pharmacy_rate_walkable[1], 'Pharmacy Vaccination Walkable rate HPI2': pharmacy_rate_walkable[2],
                    #  'Pharmacy Vaccination Walkable rate HPI3': pharmacy_rate_walkable[3],
                     'Chain Vaccination Walkable': chain_vaccination_walkable[0], 
                     'Chain Vaccination Walkable HPI1': chain_vaccination_walkable[1], 'Chain Vaccination Walkable HPI2': chain_vaccination_walkable[2], 
                     'Chain Vaccination Walkable HPI3': chain_vaccination_walkable[3],
                    #  'Chain Vaccination Walkable rate HPI1': chain_rate_walkable[1], 'Chain Vaccination Walkable rate HPI2': chain_rate_walkable[2],
                    #  'Chain Vaccination Walkable rate HPI3': chain_rate_walkable[3],
                     'Average distance': total_avg_dist[0],
                     'Average distance HPI1': total_avg_dist[1], 'Average distance HPI2': total_avg_dist[2],
                     'Average distance HPI3': total_avg_dist[3]}

    return chain_summary



# ====================================================================================



def compute_utilization_randomFCFS(K, R, z, block, locs, dists, assignment, pharmacy_locations, chain_locations, path):

    '''
    Compute the average utilization rate per store
    Import location file, export a store level result
    '''

    all_locations = pd.concat([pharmacy_locations, chain_locations])
    all_locations['Selected'] = z
    all_locations['Store_ID'] = range(1, len(all_locations) + 1)

    # vaccinations adminstrated
    all_locations['Vaccinations'] = all_locations.apply(compute_assignment_sum, axis=1, args=(locs, assignment))
    all_locations['Utilization'] = np.round(all_locations['Vaccinations'] / K, 2)

    # export
    if R is not None: all_locations.to_csv(f'{path}/Store_Results_R{R}.csv', encoding='utf-8', index=False, header=True)
    else: all_locations.to_csv(f'{path}/Store_Results.csv', encoding='utf-8', index=False, header=True)
    print('Store Results exported\n')


    return 



def compute_assignment_sum(row, locs, assignment):
    # remember that locs does not match
    return np.sum(assignment[locs == row['Store_ID']])




def compute_utilization_MIP(Scenario, Model, Chain_type, M, K, opt_constr, stage, CA_TRACT, mat_y_hpi, z, F_DH, C, C_walkable, pharmacy_locations, chain_locations, num_current_stores, num_total_stores):

    '''
    Subject to modification
    '''


    ### Stores selected
    Current['Selected_Current_Dist'] = z_current
    Current['Selected_Current_DistHPI'] = z_current_hpi
    Current['Selected_Total_Dist'] = z_total[0:num_current_stores]
    Current['Selected_Total_DistHPI'] = z_total_hpi[0:num_current_stores]

    Dollar['Selected_Total_Dist'] = z_total[num_current_stores:num_total_stores]
    Dollar['Selected_Total_DistHPI'] = z_total_hpi[num_current_stores:num_total_stores]

    # Population
    pop1 = Population[CA_TRACT['HPIQuartile'] == 1]
    pop2 = Population[CA_TRACT['HPIQuartile'] == 2]
    pop3 = Population[CA_TRACT['HPIQuartile'] == 3]
    pop4 = Population[CA_TRACT['HPIQuartile'] == 4]

    # Dollar
    Dollar['Utilization_HPI1_Total_Dist'] = np.multiply(F_DH_total, mat_y_total)[CA_TRACT['HPIQuartile'] == 1][:,num_current_stores:num_total_stores].T @ pop1 / gamma
    Dollar['Utilization_HPI1_Total_DistHPI'] = np.multiply(F_DH_total, mat_y_total_hpi)[CA_TRACT['HPIQuartile'] == 1][:,num_current_stores:num_total_stores].T @ pop1 / gamma

    Dollar['Utilization_HPI2_Total_Dist'] = np.multiply(F_DH_total, mat_y_total)[CA_TRACT['HPIQuartile'] == 2][:,num_current_stores:num_total_stores].T @ pop2 / gamma
    Dollar['Utilization_HPI2_Total_DistHPI'] = np.multiply(F_DH_total, mat_y_total_hpi)[CA_TRACT['HPIQuartile'] == 2][:,num_current_stores:num_total_stores].T @ pop2 / gamma

    Dollar['Utilization_HPI3_Total_Dist'] = np.multiply(F_DH_total, mat_y_total)[CA_TRACT['HPIQuartile'] == 3][:,num_current_stores:num_total_stores].T @ pop3 / gamma
    Dollar['Utilization_HPI3_Total_DistHPI'] = np.multiply(F_DH_total, mat_y_total_hpi)[CA_TRACT['HPIQuartile'] == 3][:,num_current_stores:num_total_stores].T @ pop3 / gamma

    Dollar['Utilization_HPI4_Total_Dist'] = np.multiply(F_DH_total, mat_y_total)[CA_TRACT['HPIQuartile'] == 4][:,num_current_stores:num_total_stores].T @ pop4 / gamma
    Dollar['Utilization_HPI4_Total_DistHPI'] = np.multiply(F_DH_total, mat_y_total_hpi)[CA_TRACT['HPIQuartile'] == 4][:,num_current_stores:num_total_stores].T @ pop4 / gamma


    return







def export_dist(path, Model, Chain, M, K, R, z, block, locs, dists, assignment, chain_locations, num_current_stores, num_total_stores, num_copies = 5):

    '''
    Export distance for visualization
    The output is in the format of (hpi, locs: pharmacy/chain, assignment, dist)
    
    block_id: with column 'market_ids', 'hpi_quantile'
    market id is the corresponding zip
    '''

    # make five copy of market id & hpi quantile
    hpi_quantile = block.hpi_quantile
    hpi_quantile_array = np.column_stack([hpi_quantile] * num_copies)
    hpi_quantile_flatten = hpi_quantile_array.flatten()

    selected_chains = chain_locations[z[num_current_stores:] == 1]
    R_chains = selected_chains.shape[0]
    locs_phar_dollar = np.where(locs <= num_current_stores - R_chains, 1, 0) 
    locs_phar_dollar_flatten = locs_phar_dollar.flatten()

    dists_flatten = dists.flatten()
    assignment_flatten = assignment.flatten()

    data = {'HPI': hpi_quantile_flatten, 'Pharmacy': locs_phar_dollar_flatten, 'Distance': np.exp(dists_flatten), 'Assignment': assignment_flatten}
    df = pd.DataFrame(data)
    if R is not None: df.to_csv(f'{path}/Distance_HPI_R{R}.csv', encoding='utf-8', index=False, header=True)
    else: df.to_csv(f'{path}/Distance_HPI.csv', encoding='utf-8', index=False, header=True)

    return 









