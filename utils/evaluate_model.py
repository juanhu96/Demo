#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21, 2022
@Author: Jingyuan Hu 
"""

import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum



##############################################################################
##############################################################################
##############################################################################

def evaluate_rate(scenario, constraint, z, pc, pf, ncp, p, closest, K,
                  num_current_stores, num_total_stores, num_tracts, 
                  scale_factor, path, MIPGap = 1e-3):
    
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
    # m.Params.IntegralityFocus = 1
    m.Params.MIPGap = MIPGap
    
    total_demand = sum(p)
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
    
    np.savetxt(path + 'y_' + scenario + '_eval_' +  constraint + '.csv', y_soln, delimiter=",")
 
    ### Finished all ###
    m.dispose()
    

    