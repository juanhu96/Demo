#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu

Temp file for any short test...
Check selected locations
"""

import os
import numpy as np
import pandas as pd
# try:
#     from demand_utils import demest_funcs as de
# except:
#     from Demand.demand_utils import demest_funcs as de

maindir = '/export/storage_covidvaccine/'
datadir='/export/storage_covidvaccine/Data/'
resultdir='/export/storage_covidvaccine/Result'

Model = 'MaxVaxHPIDistBLP'
M, K = 5, 10000
Chain_type = 'Dollar'
opt_constr = 'vaccinated'
num_current_stores = 4035 # just being lazy
nsplits = 4 # number of HPI groups


# ====================================================================================


'''
path_one = f'{maindir}/Result/MaxVaxHPIDistBLP/M{str(M)}_K{str(K)}/{Chain_type}/{opt_constr}/'
path_two = f'{maindir}/Result/MaxVaxDistBLP/M{str(M)}_K{str(K)}/{Chain_type}/{opt_constr}/'
z_one = np.genfromtxt(f'{path_one}z_total.csv', delimiter = ",", dtype = float)
z_two = np.genfromtxt(f'{path_two}z_total.csv', delimiter = ",", dtype = float)

print('Difference in the pharmacies replaced ' + str(np.sum(z_one[0 : num_current_stores] != z_two[0 : num_current_stores])))
print('Difference in the dollar stores opened ' + str(np.sum(z_one[num_current_stores:] != z_two[num_current_stores:])))


pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'zip_code', 'StateID'])
pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
pharmacy_locations.drop(columns=['StateID'], inplace=True)
pharmacy_locations['zip_code'] = pharmacy_locations['zip_code'].astype("string")

chain_locations = pd.read_csv(f"{datadir}/Raw/Location/01_DollarStores.csv", usecols=['Latitude', 'Longitude', 'Zip_Code', 'State'])
chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude', 'Zip_Code': 'zip_code'}, inplace=True)
chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
chain_locations.drop(columns=['State'], inplace=True)
chain_locations['zip_code'] = chain_locations['zip_code'].astype("string")

all_locations = pd.concat([pharmacy_locations, chain_locations])


df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv", usecols=['zip', 'hpi'])
df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)
df.rename(columns = {"zip": "zip_code"}, inplace = True)
df['zip_code']=df['zip_code'].astype("string")

chain_locations = chain_locations.merge(df[['zip_code', 'hpi_quantile']], on='zip_code', how='left')
all_locations = all_locations.merge(df[['zip_code', 'hpi_quantile']], on='zip_code', how='left')

selected_locations_one = all_locations[z_one == 1]
selected_locations_two = all_locations[z_two == 1]
print(selected_locations_one['hpi_quantile'].value_counts(), selected_locations_two['hpi_quantile'].value_counts())


selected_locations_one = chain_locations[z_one[num_current_stores:] == 1]
selected_locations_two = chain_locations[z_two[num_current_stores:] == 1]
print(selected_locations_one['hpi_quantile'].value_counts(), selected_locations_two['hpi_quantile'].value_counts())
'''


# ====================================================================================


# tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") #from prep_tracts.py
# splits = np.linspace(0, 1, nsplits+1)
# tract_hpi['hpi_quantile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1
# print(tract_hpi.head())
# blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) #from block_cw.py

# agent_data_read = agent_data_read.merge(blk_tract_cw, on='blkid', how='left')
# agent_data_read = agent_data_read.merge(tract_hpi[['tract', 'hpi_quantile']], on='tract', how='left')


# ====================================================================================

Chain = 'Dollar'
K = '8000'
path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef/{Chain}/{opt_constr}/'


z_100 = np.genfromtxt(f'{path}/z_total_fixR100.csv', delimiter = ",", dtype = float)
z_200 = np.genfromtxt(f'{path}/z_total_fixR200.csv', delimiter = ",", dtype = float)
print(np.array_equal(z_100, z_200))

# ====================================================================================

# blk_100 = pd.read_csv(f'{path}ca_blk_Dollar_dist_total_fixR100.csv', dtype={'locid': int, 'blkid': int})
# blk_200 = pd.read_csv(f'{path}ca_blk_Dollar_dist_total_fixR200.csv', dtype={'locid': int, 'blkid': int})
# print(blk_100.equals(blk_200))
# print(np.array_equal(blk_100, blk_200))

# ====================================================================================

pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
pharmacy_locations.drop(columns=['StateID'], inplace=True)

chain_locations = pd.read_csv(f"{datadir}/Raw/Location/01_DollarStores.csv", usecols=['Latitude', 'Longitude', 'State'])
chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
chain_locations.drop(columns=['State'], inplace=True)  

all_locations = pd.concat([pharmacy_locations, chain_locations])

selected_locations_100 = all_locations[z_100 == 1]
selected_locations_200 = all_locations[z_200 == 1]
print(selected_locations_100.equals(selected_locations_200))

selected_locations_100.to_csv(f'{resultdir}/Sensitivity_results/selected_locations_100.csv', encoding='utf-8', index=False, header=True)
selected_locations_200.to_csv(f'{resultdir}/Sensitivity_results/selected_locations_200.csv', encoding='utf-8', index=False, header=True)

# ====================================================================================

# assignment_100 = np.genfromtxt(f'{path}assignment_{K}_{Chain_type}_fixR100.csv', delimiter = "")
# assignment_200 = np.genfromtxt(f'{path}assignment_{K}_{Chain_type}_fixR200.csv', delimiter = "")
# assignment_400 = np.genfromtxt(f'{path}assignment_{K}_{Chain_type}_fixR400.csv', delimiter = "")
# print(np.array_equal(assignment_100, assignment_200), np.array_equal(assignment_100, assignment_400))