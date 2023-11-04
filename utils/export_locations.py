#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu

Export selected locations
python3 export_locations.py Dollar 8000 5 4 True 100
"""

import os
import numpy as np
import pandas as pd

import sys 
Chain = sys.argv[1] # Chain
K = int(sys.argv[2]) # K
M = int(sys.argv[3]) # M
nsplits = int(sys.argv[4])
capcoef = sys.argv[5]
R = sys.argv[6] # R



def export_locations(Chain, M, K, nsplits, capcoef, R, opt_constr='vaccinated', datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result'):

    Model = 'MaxVaxHPIDistBLP'
    path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef/{Chain}/{opt_constr}/'
    z_BLP = np.genfromtxt(f'{path}/z_total_fixR{R}.csv', delimiter = ",", dtype = float)

    Model = 'MaxVaxDistLogLin'
    path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q_capcoef/{Chain}/{opt_constr}/'
    z_LogLin = np.genfromtxt(f'{path}/z_total_fixR{R}.csv', delimiter = ",", dtype = float)

    # ====================================================================================

    pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
    pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
    pharmacy_locations.drop(columns=['StateID'], inplace=True)

    chain_locations = pd.read_csv(f"{datadir}/Raw/Location/01_DollarStores.csv", usecols=['Latitude', 'Longitude', 'State'])
    chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
    chain_locations.drop(columns=['State'], inplace=True)  

    # ====================================================================================

    BLP_pharmacy = pharmacy_locations[z_BLP[:len(pharmacy_locations)] == 0] # closed
    BLP_chain = chain_locations[z_BLP[len(pharmacy_locations):] == 1]

    LogLin_pharmacy = pharmacy_locations[z_LogLin[:len(pharmacy_locations)] == 0] # closed
    LogLin_chain = chain_locations[z_LogLin[len(pharmacy_locations):] == 1]

    BLP_pharmacy.to_csv(f'{resultdir}/Locations/BLP_pharmacy_R{R}.csv', encoding='utf-8', index=False, header=True)
    BLP_chain.to_csv(f'{resultdir}/Locations/BLP_chain_R{R}.csv', encoding='utf-8', index=False, header=True)

    LogLin_pharmacy.to_csv(f'{resultdir}/Locations/LogLin_pharmacy_R{R}.csv', encoding='utf-8', index=False, header=True)
    LogLin_chain.to_csv(f'{resultdir}/Locations/LogLin_chain_R{R}.csv', encoding='utf-8', index=False, header=True)

    return



if __name__ == "__main__":
    export_locations(Chain, M, K, nsplits, capcoef, int(R))