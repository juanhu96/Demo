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
try:
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import demest_funcs as de

maindir = '/export/storage_covidvaccine/'
datadir='/export/storage_covidvaccine/Data/'

Model = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP']
M, K = 5, 10000
Chain_type = 'Dollar'
opt_constr = 'assigned'
num_current_stores = 4035 # just being lazy


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
nsplits = 3 # number of HPI groups
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


# ====================================================================================






