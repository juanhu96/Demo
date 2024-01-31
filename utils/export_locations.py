#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu

Export selected locations
python3 export_locations.py Dollar 8000 5 4 True 100
"""

import os
import sys 
import numpy as np
import pandas as pd


#=================================================================
# SETTINGS

K = int(sys.argv[1])
M = int(sys.argv[2])
nsplits = int(sys.argv[3])

capcoef = any(['capcoef' in arg for arg in sys.argv])
mnl = any([arg == 'mnl' for arg in sys.argv])
logdist_above = any(['logdistabove' in arg for arg in sys.argv])
if logdist_above:
    logdist_above_arg = [arg for arg in sys.argv if 'logdistabove' in arg][0]
    logdist_above_thresh = float(logdist_above_arg.replace('logdistabove', ''))
else: logdist_above_thresh = 0

flexible_consideration = any(['flex' in arg for arg in sys.argv])
flex_thresh = dict(zip(["urban", "suburban", "rural"], [2,3,15]))

replace = any(['replace' in arg for arg in sys.argv])
if replace:
    replace_arg = [arg for arg in sys.argv if 'replace' in arg][0]
    R = int(replace_arg.replace('replace', ''))
else: R = None

add = any(['add' in arg for arg in sys.argv])
if add:
    add_arg = [arg for arg in sys.argv if 'add' in arg][0]
    A = int(add_arg.replace('add', ''))
else: A = None

setting_tag = f'_{str(K)}_1_{nsplits}q' if flexible_consideration else f'_{str(K)}_{M}_{nsplits}q' 
setting_tag += '_capcoefs0' if capcoef else ''
setting_tag += "_mnl" if mnl else ""
setting_tag += "_flex" if flexible_consideration else ""
setting_tag += f"thresh{str(list(flex_thresh.values())).replace(', ', '_').replace('[', '').replace(']', '')}" if flexible_consideration else ""
setting_tag += f"_logdistabove{logdist_above_thresh}" if logdist_above else ""
setting_tag += f"_R{R}" if replace else ""
setting_tag += f"_A{A}" if add else ""


# ================================================================================

def main(M, K, nsplits, setting_tag):

    print(f"Start export locations under setting {setting_tag}...\n")
    Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistLogLin', 'MNL_partial']

    for Model in Model_list:
        export_locations(Model=Model,
                         Chain='Dollar',
                         M=M,
                         K=K,
                         nsplits=nsplits,
                         setting_tag=setting_tag)





def export_locations(Model, Chain, M, K, nsplits, setting_tag, suffix='', opt_constr='vaccinated', datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result'):

    Model_name_dict = {'MaxVaxHPIDistBLP': 'Assignment_BLP', 'MaxVaxDistLogLin': 'Assignment_LogLin', 'MNL_partial': 'Choice_BLP'}

    path = f'{resultdir}/{Model}/M{str(M)}_K{str(K)}_{nsplits}q/{Chain}/{opt_constr}/'
    z = np.genfromtxt(f'{path}/z_total{setting_tag}.csv', delimiter = ",", dtype = float)

    # ====================================================================================

    pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
    pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
    pharmacy_locations.drop(columns=['StateID'], inplace=True)

    chain_locations = pd.read_csv(f"{datadir}/Raw/Location/01_DollarStores.csv", usecols=['Latitude', 'Longitude', 'State'])
    chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
    chain_locations.drop(columns=['State'], inplace=True)  

    # ====================================================================================

    model_pharmacy = pharmacy_locations[z[:len(pharmacy_locations)] == 0] # closed
    model_chain = chain_locations[z[len(pharmacy_locations):] == 1]

    model_pharmacy.to_csv(f'{resultdir}/Locations/{Model_name_dict[Model]}_pharmacy{setting_tag}.csv', encoding='utf-8', index=False, header=True)
    model_chain.to_csv(f'{resultdir}/Locations/{Model_name_dict[Model]}_chain{setting_tag}.csv', encoding='utf-8', index=False, header=True)

    return



if __name__ == "__main__":
    main(M, K, nsplits, setting_tag)