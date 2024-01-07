#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
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



def optimize_rate(scenario, constraint, pc, pf, ncp, p, K, closest,
                  num_current_stores, num_total_stores, num_tracts, 
                  scale_factor, path, R = None, heuristic=False, MIPGap = 1e-3):
    
    """
    Parameters
    ----------
    scenario : string
        "current": current stores only
        "total": current and dollar stores
    
    constraint : string
        'assigned', 'vaccinated'
        whether the capacity constraint is based on assignments or vaccinations

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
    

    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'y')
    
    
    ### Objective ###
    m.setObjective(quicksum(pf[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    
    
    ### Constraints ###
    if constraint == 'assigned':
        for j in range(num_stores):
            m.addConstr(quicksum(p[i] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
    elif constraint == 'vaccinated':
        for j in range(num_stores):
            m.addConstr(quicksum(pf[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])

    for i in range(num_tracts):
        m.addConstr(quicksum(y[i * num_stores + j] for j in range(num_stores)) <= 1)
        
    for k in range(num_tracts * num_stores):
        m.addConstr(y[k] <= closest[k])

    m.addConstr(z.sum() == num_current_stores, name = 'N')

    if R is not None: m.addConstr(quicksum(z[j] for j in range(num_current_stores)) == num_current_stores - R)

    ## Solve ###
    m.update()
    start = time.time()
    m.optimize()
    end = time.time()

    ### Export ###
    z_soln = np.zeros(num_stores)
    for j in range(num_stores):
        z_soln[j] = z[j].X
    
    y_soln = np.zeros(num_tracts * num_stores)
    for k in range(num_tracts * num_stores):
        y_soln[k] = y[k].X    
    
    
    ### Summary ###
    z_file_name = f'{path}z_{scenario}'
    y_file_name = f'{path}y_{scenario}'
    
    if R is not None: 
        z_file_name += f'_fixR{str(R)}'
        y_file_name += f'_fixR{str(R)}'

    if heuristic:
        z_file_name += '_heuristic'
        y_file_name += '_heuristic'

    np.savetxt(f'{z_file_name}.csv', z_soln, delimiter=",")
    np.savetxt(f'{y_file_name}.csv', y_soln, delimiter=",")

 
    ### Finished all ###
    m.dispose()



##############################################################################  
##############################################################################
##############################################################################   



def optimize_rate_MNL(scenario, pf, v, C, K, num_current_stores, num_total_stores, num_tracts, scale_factor, path, R=None, MIPGap = 5e-2):
    
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
    # m.Params.IntegralityFocus = 1
    m.Params.MIPFocus = 3 # to focus on the bound
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
        # m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in range(num_stores)) == 1)
        m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == 1)
        
    for j in range(num_stores):
        m.addConstr(quicksum(pf[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])

    m.addConstr(z.sum() == num_current_stores, name = 'N')
    if R is not None: m.addConstr(quicksum(z[j] for j in range(num_current_stores)) == num_current_stores - R)

    for i in range(num_tracts):
        for j in C[i]:
            m.addConstr(x[i] - y[i * num_stores + j] <= 1 - z[j])
            m.addConstr(y[i * num_stores + j] <= x[i])
            m.addConstr((1 + v[i * num_stores + j]) * y[i * num_stores + j] <= z[j])


    print("****************** FINISHED CONSTRUCTING, START OPTIMIZING ******************\n")


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
    z_file_name = f'{path}z_{scenario}'
    
    if R is not None: 
        z_file_name += f'_fixR{str(R)}'

    np.savetxt(f'{z_file_name}.csv', z_soln, delimiter=",")


    ### Finished all ###
    m.dispose()



##############################################################################  
##############################################################################
############################################################################## 



def optimize_rate_MNL_new(scenario, pf, v, C, K, num_current_stores, num_total_stores, num_tracts, scale_factor, path, R=None, MIPGap = 5e-2):

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("Vaccination")
    # m.Params.IntegralityFocus = 1
    m.Params.MIPFocus = 3 # to focus on the bound
    m.Params.MIPGap = MIPGap
    

    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, name = 'y')
    x = m.addVars(num_tracts, lb = 0, name = 'x')

    T = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'T')
    t = m.addVars(num_stores, lb = 0, ub = 1, name = 't')
    
    ### Objective ###

    # NOTE: pf is now p_i * v_ij
    m.setObjective(quicksum(pf[k] * T[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    

    ### Constraints ###
    for i in range(num_tracts):
        # m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in range(num_stores)) == 1)
        m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == 1)

    for j in range(num_stores):
        m.addConstr(quicksum(pf[i * num_stores + j] * T[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
        m.addConstr(t[j] <= z[j])

    m.addConstr(z.sum() == num_current_stores, name = 'N')
    if R is not None: m.addConstr(quicksum(z[j] for j in range(num_current_stores)) == num_current_stores - R)

    for i in range(num_tracts):
        for j in C[i]:
            m.addConstr(x[i] - y[i * num_stores + j] <= 1 - z[j])
            m.addConstr(y[i * num_stores + j] <= x[i])
            m.addConstr((1 + v[i * num_stores + j]) * y[i * num_stores + j] <= z[j])

            m.addConstr(T[i * num_stores + j] <= t[j])
            m.addConstr(T[i * num_stores + j] <= y[i * num_stores + j])
            m.addConstr(T[i * num_stores + j] >= t[j] + y[i * num_stores + j] - 1)
            

    print("****************** FINISHED CONSTRUCTING, START OPTIMIZING ******************\n")


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
    z_file_name = f'{path}z_{scenario}'
    
    if R is not None: 
        z_file_name += f'_fixR{str(R)}'

    np.savetxt(f'{z_file_name}_new.csv', z_soln, delimiter=",")


    ### Finished all ###
    m.dispose()



##############################################################################  
##############################################################################
##############################################################################   
    


def optimize_dist(scenario, pc, ncp, p, K,
                  num_current_stores, num_total_stores, num_tracts,
                  scale_factor, path, MIPGap = 5e-2):
    
    '''
    See optimize_rate()
    '''
    

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("Vaccination")
    # m.Params.IntegralityFocus = 1
    m.Params.MIPGap = MIPGap
    

    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores
    
    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'y')
    
    
    ### Objective ###
    m.setObjective(quicksum(pc[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MINIMIZE)
    
    ### Constraints ###
    for j in range(num_stores):
        m.addConstr(quicksum(p[i] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
    
    for i in range(num_tracts):
        m.addConstr(quicksum(y[i * num_stores + j] for j in range(num_stores)) == 1) ## strictly equal
    
    m.addConstr(z.sum() == num_current_stores, name = 'N')    
        

    ### Solve ###
    m.update()
    start = time.time()
    m.optimize()
    end = time.time()

    ### Export ###
    z_soln = np.zeros(num_stores)
    for j in range(num_stores):
            z_soln[j] = z[j].X

    y_soln = np.zeros(num_tracts * num_stores)
    for k in range(num_tracts * num_stores):
            y_soln[k] = y[k].X    
    
    
    ### Summary ###
    result_df = pd.DataFrame([sum(z_soln), round(end - start,1)], index =['Stores used', 'Time'], columns =['Value'])
    np.savetxt(f'{path}z_{scenario}.csv', z_soln, delimiter=",")
    np.savetxt(f'{path}y_{scenario}.csv', y_soln, delimiter=",")
    result_df.to_csv(f'{path}result_{scenario}.csv') 
            
     
    ### Finished all ###
    m.dispose()


    
##############################################################################  
##############################################################################
##############################################################################       



def optimize_rate_fix(scenario, constraint, ncp, pv, p, closest, K,
                       num_current_stores, num_total_stores, num_tracts, 
                       scale_factor, path, MIPGap = 5e-3):
    '''
    Parameters
    ----------
    
    See optimize_rate(), except
    
    pf : array
        scaled population * willingness
        
    ncp : array
        n copies of population vector
    
    pv : array
        scaled population * fix willingness (vaccination rate) 
    
    '''


    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("Vaccination")
    m.Params.IntegralityFocus = 1
    m.Params.MIPGap = MIPGap
    

    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores
    
    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'y')
    
    
    ### Objective ###
    m.setObjective(quicksum(pv[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    
    
    ### Constraints ###

    if constraint == 'assigned':
        for j in range(num_stores):
            m.addConstr(quicksum(p[i] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
    elif constraint == 'vaccinated':
        for j in range(num_stores):
            m.addConstr(quicksum(pv[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j]) # TODO: double check the formula
        
    for i in range(num_tracts):
        m.addConstr(quicksum(y[i * num_stores + j] for j in range(num_stores)) <= 1)
    
    m.addConstr(z.sum() == num_current_stores, name = 'N')    
       
    for k in range(num_tracts * num_stores):
        m.addConstr(y[k] <= closest[k])


    ### Solve ###
    m.update()
    start = time.time()
    m.optimize()
    end = time.time()


    ### Export ###
    z_soln = np.zeros(num_stores)
    for j in range(num_stores):
            z_soln[j] = z[j].X

    y_soln = np.zeros(num_tracts * num_stores)
    for k in range(num_tracts * num_stores):
            y_soln[k] = y[k].X    
    
    
    ### Summary ###
    result_df = pd.DataFrame([sum(z_soln), round(end - start,1)], index =['Stores used', 'Time'], columns =['Value'])
    np.savetxt(f'{path}z_{scenario}.csv', z_soln, delimiter=",")
    np.savetxt(f'{path}y_{scenario}.csv', y_soln, delimiter=",")
    result_df.to_csv(f'{path}result_{scenario}.csv') 

    ### Finished all ###
    m.dispose()    
    
    
 
    
 
    
 
    
 
    
 
    