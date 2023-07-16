#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import pandas as pd
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# from daa_helpers import



def main():

    Population, Quartile, C_current_mat, C_total_mat, num_tracts, num_current_stores, num_total_stores = import_location()
    
    C_current_order = [[sorted(row).index(x) for x in row] for row in C_current]


    daa(Population, C_current_order)










def daa(P, C_order, V, num_tracts, num_stores, M=5, K = 8000)

    """
    Run deferred acceptence algorithm with M closest sites
    """

    S = [8000 for i in range(num_stores)] # initial capacity


    for t in range(M):
        print(f'Assigning tracts to their {str(t+1)}th site...\n')


        # for each site, get all tracts which it is their tth closest
        for j in range(num_stores):
            candidate_sites = C_order[i, t] == j

        # the demand
        arrivals = sum([P[i] for i in candidate_sites])

        if arrivals <= S[j]:
            
            S[j] = S[j] - arrivals
            P[i] = P[i] - ... for i in candidate_sites

        else:







if __name__ == "__main__":
    main()
