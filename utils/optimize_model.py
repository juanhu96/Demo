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


def optimize_rate_MNL(scenario,
                    pv,
                    v,
                    C,
                    K,
                    R,
                    A,
                    closest,
                    num_current_stores,
                    num_total_stores,
                    num_tracts,
                    scale_factor,
                    setting_tag,
                    path,
                    MIPGap=5e-2):
    
    """
    Parameters
    ----------
    scenario : string
        "current": current stores only
        "total": current and dollar stores
    
    pv : array
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
    m.Params.TimeLimit = 21600 # 6 hours

    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, name = 'y')
    x = m.addVars(num_tracts, lb = 0, name = 'x')

    
    ### Objective ###

    # NOTE: pv is now p_i * v_ij
    m.setObjective(quicksum(pv[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    

    ### Constraints ###
    for i in range(num_tracts):
        m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in range(num_stores)) == 1)
        # m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == 1)
        
    for j in range(num_stores):
        m.addConstr(quicksum(pv[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])

    # for k in range(num_tracts * num_stores):
    #     m.addConstr(y[k] <= closest[k])

    if A is not None:
        print(f"Keep all current locations and add {A} locations")
        m.addConstr(z.sum() == num_current_stores + A, name = 'N')
        m.addConstr(quicksum(z[i] for i in range(num_current_stores)) == num_current_stores)
    else:
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
    np.savetxt(f'{z_file_name}{setting_tag}.csv', z_soln, delimiter=",")


    ### Finished all ###
    m.dispose()



##############################################################################  
##############################################################################
############################################################################## 


def optimize_rate_MNL_partial(scenario,
                            pv,
                            v,
                            C,
                            closest,
                            K,
                            R,
                            A,
                            num_current_stores,
                            num_total_stores,
                            num_tracts,
                            path,
                            setting_tag,
                            scale_factor,
                            MIPGap=5e-2):


    """
    Parameters
    ----------
    scenario : string
        "current": current stores only
        "total": current and dollar stores
    
    pv : array
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
    m.Params.MIPFocus = 3 # to focus on the bound
    m.Params.TimeLimit = 21600 # 6 hours
    # small problems require higher accuracy (also easier to solve)
    if (A is not None) or (R is not None): 
        m.Params.MIPGap = 1e-3
    else: 
        m.Params.MIPGap = MIPGap
    
    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, name = 'y')
    x = m.addVars(num_tracts, lb = 0, name = 'x')

    T = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'T')
    # t = m.addVars(num_stores, lb = 0, ub = 1, name = 't') # proportion allocation
    t = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 't') # priority allocation

    
    ### Objective ###
    # pv is now p_i * v_ij
    m.setObjective(quicksum(pv[k] * T[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    

    ### Constraints ###
    for i in range(num_tracts):
        # m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in range(num_stores)) == 1)
        m.addConstr(x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == 1)

    for j in range(num_stores):
        m.addConstr(quicksum(pv[i * num_stores + j] * T[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
        # m.addConstr(t[j] <= z[j])
        m.addConstrs(t[i * num_stores + j] <= z[j] for i in range(num_tracts))

    if A is not None:
        print(f"Keep all current locations and add {A} locations")
        m.addConstr(z.sum() == num_current_stores + A, name = 'N')
        m.addConstr(quicksum(z[i] for i in range(num_current_stores)) == num_current_stores)
    else:
        m.addConstr(z.sum() == num_current_stores, name = 'N')

    if R is not None: 
        print(f"Repalce {R} locations only\n")
        m.addConstr(quicksum(z[j] for j in range(num_current_stores)) == num_current_stores - R)

    for i in range(num_tracts):
        for j in C[i]:
            m.addConstr(x[i] - y[i * num_stores + j] <= 1 - z[j])
            m.addConstr(y[i * num_stores + j] <= x[i])
            m.addConstr((1 + v[i * num_stores + j]) * y[i * num_stores + j] <= z[j])

            # m.addConstr(T[i * num_stores + j] <= t[j])
            # m.addConstr(T[i * num_stores + j] <= y[i * num_stores + j])
            # m.addConstr(T[i * num_stores + j] >= t[j] + y[i * num_stores + j] - 1)
            m.addConstr(T[i * num_stores + j] <= t[i * num_stores + j])
            m.addConstr(T[i * num_stores + j] <= y[i * num_stores + j])
            m.addConstr(T[i * num_stores + j] >= t[i * num_stores + j] + y[i * num_stores + j] - 1)
            
    for k in range(num_tracts * num_stores):
        m.addConstr(t[k] <= closest[k])

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
    np.savetxt(f'{z_file_name}{setting_tag}.csv', z_soln, delimiter=",")


    ### Finished all ###
    m.dispose()



##############################################################################  
##############################################################################
##############################################################################  



def optimize_rate_MNL_partial_new(scenario,
                            pg,
                            v,
                            C,
                            closest,
                            K,
                            big_M,
                            R,
                            A,
                            num_current_stores,
                            num_total_stores,
                            num_tracts,
                            path,
                            setting_tag,
                            scale_factor,
                            MIPGap=5e-2,
                            FeasibilityTol=1e-4):


    """
    Parameters
    ----------
    scenario : string
        "current": current stores only
        "total": current and dollar stores
    
    pv : array
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
    m.Params.MIPFocus = 3 # to focus on the bound
    m.Params.TimeLimit = 21600 # 6 hours
    # m.Params.TimeLimit = 36000 # 10 hours FOR M = 10
    # m.Params.FeasibilityTol = FeasibilityTol # tolerence
    # small problems require higher accuracy (also easier to solve)
    if (A is not None) or (R is not None): 
        m.Params.MIPGap = 1e-3
    else: 
        m.Params.MIPGap = MIPGap
    
    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores

    print("============== START BUILDING ================\n")
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, name = 'y')
    x = m.addVars(num_tracts, lb = 0, name = 'x')

    T = m.addVars(num_tracts * num_stores, lb = 0, name = 'T') # since w could be greater than 1
    t = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 't') # priority allocation
    B = m.addVars(num_tracts, vtype=GRB.BINARY, name = 'B')  # DIFF B VERSION 3
    
    ### Objective ###
    # pv is now p_i * v_ij
    m.setObjective(quicksum(pg[k] * T[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE) # DIFF A 
    

    ### Constraints ###
    for i in range(num_tracts):
        # m.addConstr(quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == 1) # DIFF B VERSION 1
        # m.addConstr(epi[i] * x[i] + quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == 1) # DIFF B VERSION 2
        m.addConstr(quicksum(v[i * num_stores + j] * y[i * num_stores + j] for j in C[i]) == B[i]) # DIFF B
        m.addConstr(x[i] <= big_M[i] * B[i]) # DIFF B
        # m.addConstr(B[i] <= quicksum(z[j] for j in C[i])) # DIFF C
        m.addConstrs(B[i] >= z[j] for j in C[i]) # DIFF C

    for j in range(num_stores):
        m.addConstr(quicksum(pg[i * num_stores + j] * T[i * num_stores + j] for i in range(num_tracts)) <= K * z[j]) # DIFF D
        m.addConstrs(t[i * num_stores + j] <= z[j] for i in range(num_tracts)) # y_rs < z_s # DIFF I

    if A is not None:
        print(f"Keep all current locations and add {A} locations")
        m.addConstr(z.sum() == num_current_stores + A, name = 'N')
        m.addConstr(quicksum(z[i] for i in range(num_current_stores)) == num_current_stores)
    else:
        m.addConstr(z.sum() == num_current_stores, name = 'N')

    if R is not None: 
        print(f"Repalce {R} locations only\n")
        m.addConstr(quicksum(z[j] for j in range(num_current_stores)) == num_current_stores - R)

    for i in range(num_tracts):
        for j in C[i]:

            # m.addConstr(epi[i] * (x[i] - y[i * num_stores + j]) <= 1 - z[j]) # DIFF F EPSILON
            # m.addConstr(y[i * num_stores + j] <= x[i]) # DIFF G EPSILON
            # m.addConstr((1 + v[i * num_stores + j]) * y[i * num_stores + j] <= z[j]) # DIFF H EPSILON

            m.addConstr(x[i] - y[i * num_stores + j] <= big_M[i] * (1 - z[j])) # DIFF F
            m.addConstr(y[i * num_stores + j] <= x[i]) # DIFF G
            m.addConstr(v[i * num_stores + j] * y[i * num_stores + j] <= z[j]) # DIFF H

            # y is w, t is y
            m.addConstr(T[i * num_stores + j] <= y[i * num_stores + j])
            m.addConstr(T[i * num_stores + j] <= big_M[i] * t[i * num_stores + j])
            m.addConstr(T[i * num_stores + j] >= y[i * num_stores + j] + big_M[i] * t[i * num_stores + j] - big_M[i])
            
    # for k in range(num_tracts * num_stores): # not necessary
    #     m.addConstr(t[k] <= closest[k])

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
    np.savetxt(f'{z_file_name}{setting_tag}.csv', z_soln, delimiter=",")


    ### Finished all ###
    m.dispose()

    return z_soln