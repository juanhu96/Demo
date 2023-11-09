#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2023
@Author: Jingyuan Hu 
"""

import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


# ============================================================================
# ============================================================================
# ============================================================================


def optimize_rate_MNL(scenario, pf, v, C, K, num_current_stores, num_total_stores, num_tracts, path, MIPGap = 1e-3):
    
    """
    Parameters
    ----------
    scenario : string
        "current": current stores only
        "total": current and dollar stores
    
    pf : array
        scaled population * v

    v : array
        flatten array of v_{ij} = e^{mu_ij}

    C : list
        list of lists of sites that is within M closest to a region
        
    K : scalar
        capacity of a single site
        
    path : string
        directory for results

    """


    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("Vaccination")
    m.Params.IntegralityFocus = 1
    m.Params.MIPGap = MIPGap
    

    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, name = 'y')
    x = m.addVars(num_tracts, lb = 0, name = 'x')

    
    ### Objective ###

    # NOTE: pf is now p_i * v_ij
    m.setObjective(quicksum(pf[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    

    ### Constraints ###
    for i in range(num_tracts):
        m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in range(num_stores)) == 1)
        
    for j in range(num_stores):
        m.addConstr(quicksum(pf[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])

    m.addConstr(z.sum() == num_current_stores, name = 'N')

    for i in range(num_tracts):
        for j in C[i]:
            m.addConstr(x[i] - y[i * num_stores + j] <= 1 - z[j])
            m.addConstr(y[i * num_stores + j] <= x[i])
            m.addConstr((1 + v[i * num_stores + j]) * y[i * num_stores + j] <= z[j])


    ## Solve ###
    m.update()
    start = time.time()
    m.optimize()
    end = time.time()

    ### Export ###
    z_soln = np.zeros(num_stores)
    for j in range(num_stores):
        z_soln[j] = z[j].X
    
    
    ### Summary ###
    np.savetxt(f'{path}z_MNL.csv', z_soln, delimiter=",")

 
    ### Finished all ###
    m.dispose()