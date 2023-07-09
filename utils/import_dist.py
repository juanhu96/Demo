#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import pandas as pd
import numpy as np


def import_dist(Chain_type, M, datadir="/export/storage_covidvaccine/Data", MAXDIST = 10000000):

    ### Census Tract 
    Population = np.genfromtxt(f'{datadir}/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    Quartile = np.genfromtxt(f'{datadir}/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)
    
    ### Current ###
    C_current = np.genfromtxt(f'{datadir}/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current = C_current.astype(int)
    C_current = C_current.T
    num_tracts, num_current_stores = np.shape(C_current)

    ### Chains ###
    C_chains = np.genfromtxt(f'{datadir}/CA_dist_matrix_' + Chain_type + '.csv', delimiter = ",", dtype = float)
    C_chains = C_chains.astype(int)
    C_chains = C_chains.T
    num_tracts, num_chains_stores = np.shape(C_chains)
    C_chains = np.where(C_chains < 0, 1317574, C_chains) # avoid negative numbers for high schools
    
    ### Total ###
    C_total = np.concatenate((C_current, C_chains), axis = 1)
    num_total_stores = num_current_stores + num_chains_stores
    
    ###########################################################################
    
    ### Travel to the closest M stores only
    Closest_current = np.ones((num_tracts, num_current_stores))
    Closest_total = np.ones((num_tracts, num_total_stores))
    np.put_along_axis(Closest_current, np.argpartition(C_current,M,axis=1)[:,M:],0,axis=1)
    np.put_along_axis(Closest_total, np.argpartition(C_total,M,axis=1)[:,M:],0,axis=1)
    
    ###########################################################################

    C_currentMinDist = C_current * Closest_current
    C_totalMinDist = C_total * Closest_total
    C_currentMinDist = np.where(C_currentMinDist == 0, MAXDIST, C_currentMinDist)
    C_totalMinDist = np.where(C_totalMinDist == 0, MAXDIST, C_totalMinDist)

    ###########################################################################

    Closest_current = Closest_current.flatten()
    Closest_total = Closest_total.flatten()

    return Population, Quartile, C_current, C_chains, C_total, Closest_current, Closest_total, C_currentMinDist, C_totalMinDist, num_tracts, num_current_stores, num_chains_stores, num_total_stores

