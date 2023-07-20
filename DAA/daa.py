#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import pandas as pd
import numpy as np
import copy

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from daa_helpers import import_location, construct_V, compute_results



def main():

    Population, Quartile, C_current, C_total, num_tracts, num_current_stores, num_total_stores = import_location()
    
    Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]] # v2
    V_D_current, V_D_total, V_DH_current, V_DH_total  = construct_V(Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)


    C_current_order = np.argsort(C_current, axis=1)
    S, y = daa(Population, C_current_order, V_DH_current, num_tracts, num_current_stores)
    total_vaccination, total_proportion = compute_results(Population, Quartile, y)
    print(total_vaccination, total_proportion)


    C_total_order = np.argsort(C_total, axis=1)
    S, y = daa(Population, C_total_order, V_DH_total, num_tracts, num_total_stores)
    total_vaccination, total_proportion = compute_results(Population, Quartile, y)
    print(total_vaccination, total_proportion)




    


def daa(P, C_order, V, num_tracts, num_stores, M=10, K=10000):

    """
    Deferred acceptence algorithm with M closest sites
    """

    S = [K for i in range(num_stores)] # initial capacity
    P_m = copy.copy(P) # unassigned
    y = np.zeros((num_tracts, num_stores)) # vaccinations


    for m in range(M):

        # for each site j, get all tracts which j is their mth closest
        for j in range(num_stores):

            candidate_tracts = np.where(C_order[:, m] == j)[0] 
            arrivals = [P_m[i] * V[i][j] for i in candidate_tracts]
            total_arrivals = sum(arrivals)

            if len(candidate_tracts):

                if total_arrivals <= S[j]:

                    for i in candidate_tracts: 

                        y[i][j] = (P_m[i] * V[i][j])
                        P_m[i] = 0
                        

                    S[j] = S[j] - total_arrivals

                else:

                    for i in candidate_tracts:

                        y[i][j] = (P_m[i] * V[i][j] * S[j] / total_arrivals)
                        P_m[i] = (P_m[i] * V[i][j]) - y[i][j] # unmet visit are dropped

                    S[j] = 0


            # ISSUE: if we allocate to 3rd closest but it is already full
            # do we allocate to the 4th in the 3rd round, 
            # or still 3rd in the 3rd round and no one gets vaccinated

            # if we do the former, we would be allocating them to the 6th in the 5th round
            # which violates the M-constraint


    return S, y




if __name__ == "__main__":
    main()
