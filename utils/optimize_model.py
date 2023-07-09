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



def optimize_rate(scenario, Demand_estimation, pc, pf, ncp, p, K, closest,
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
    m.Params.IntegralityFocus = 1
    m.Params.MIPGap = MIPGap
    
    total_demand = sum(p)
    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores

    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'y')
    
    
    ### Objective ###
    m.setObjective(quicksum(pf[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    
    
    ### Constraints ###
    for j in range(num_stores):
        # m.addConstr(quicksum(ncp[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
        m.addConstr(quicksum(p[i] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])


    for i in range(num_tracts):
        m.addConstr(quicksum(y[i * num_stores + j] for j in range(num_stores)) <= 1)
    
    m.addConstr(z.sum() == num_current_stores, name = 'N')    
        
    for k in range(num_tracts * num_stores):
        m.addConstr(y[k] <= closest[k])


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
    if scenario == "current":
        store_used = sum(z_soln)
        vaccine_rate = pf @ y_soln / total_demand
        avg_dist = (pc @ y_soln / total_demand) * scale_factor    
        allocation_rate = ncp @ y_soln / total_demand

        result = [store_used, vaccine_rate, avg_dist, allocation_rate, round(end - start,1)]        
        result_df = pd.DataFrame(result, index =['Stores used', 'Vaccination rate', 'Avg distance', 'Allocation rate', 'Time'], columns =['Value'])
            
        np.savetxt(path + 'z_' + Demand_estimation + '_current''.csv', z_soln, delimiter=",")
        np.savetxt(path + 'y_' + Demand_estimation + '_current.csv', y_soln, delimiter=",")
        result_df.to_csv(path + 'result_' + Demand_estimation + '_current.csv')
        
    elif scenario == "total":
        store_used = sum(z_soln)
        vaccine_rate = (pf @ y_soln / total_demand)
        avg_dist = (pc @ y_soln / total_demand) * scale_factor
        allocation_rate = ncp @ y_soln / total_demand
                         
        num_current_store_used = sum(z_soln[0 : num_current_stores])
        num_dollar_store_used = store_used - num_current_store_used
            
        dollar_store_demand = ncp[num_current_stores * num_tracts : num_total_stores * num_tracts] @ y_soln[num_current_stores * num_tracts : num_total_stores * num_tracts]
        dollar_store_allocation_rate = dollar_store_demand / total_demand
            
        result = [store_used, vaccine_rate, avg_dist, allocation_rate, num_current_store_used, num_dollar_store_used, dollar_store_allocation_rate, round(end - start,1)]
        result_df = pd.DataFrame(result, index =['Stores used', 'Vaccination rate', 'Avg distance', 'Allocation rate',\
                                                 'Current store used', 'Dollar store used', 'Dollar store allocation rate', 'Time'], columns =['Value'])
                
            
        np.savetxt(path + 'z_' + Demand_estimation + '_total.csv', z_soln, delimiter=",")
        np.savetxt(path + 'y_' + Demand_estimation + '_total.csv', y_soln, delimiter=",")
        result_df.to_csv(path + 'result_' + Demand_estimation + '_total.csv')    

 
    ### Finished all ###
    m.dispose()
    


##############################################################################  
##############################################################################
##############################################################################   
    


def optimize_dist(scenario, pc, pf, ncp, p, K,
                  num_current_stores, num_total_stores, num_tracts,
                  scale_factor, path, MIPGap = 1e-2):
    
    '''
    See optimize_rate()
    '''
    

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("Vaccination")
    m.Params.IntegralityFocus = 1
    m.Params.MIPGap = MIPGap
    
    total_demand = sum(p)
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
    if scenario == "current":
        store_used = sum(z_soln)
        vaccine_rate = pf @ y_soln / total_demand
        avg_dist = (pc @ y_soln / total_demand) * scale_factor    
        allocation_rate = ncp @ y_soln / total_demand

        result = [store_used, vaccine_rate, avg_dist, allocation_rate, round(end - start,1)]        
        result_df = pd.DataFrame(result, index =['Stores used', 'Vaccination rate', 'Avg distance', 'Allocation rate', 'Time'],\
                                 columns =['Value'])
        
        np.savetxt(path + 'z_current.csv', z_soln, delimiter=",")
        np.savetxt(path + 'y_current.csv', y_soln, delimiter=",")
        result_df.to_csv(path + 'result_current.csv')
    
    elif scenario == "total":
        store_used = sum(z_soln)
        vaccine_rate = (pf @ y_soln / total_demand)
        avg_dist = (pc @ y_soln / total_demand) * scale_factor    
        allocation_rate = ncp @ y_soln / total_demand
                     
        num_current_store_used = sum(z_soln[0 : num_current_stores])
        num_dollar_store_used = store_used - num_current_store_used
        
        dollar_store_demand = ncp[num_current_stores * num_tracts : num_total_stores * num_tracts] @ y_soln[num_current_stores * num_tracts : num_total_stores * num_tracts]
        dollar_store_allocation_rate = dollar_store_demand / total_demand
        
        result = [store_used, vaccine_rate, avg_dist, allocation_rate, num_current_store_used, num_dollar_store_used, dollar_store_allocation_rate, round(end - start,1)]
        result_df = pd.DataFrame(result, index =['Stores used', 'Vaccination rate', 'Avg distance', 'Allocation rate',\
                                                 'Current store used', 'Dollar store used', 'Dollar store allocation rate',\
                                                     'Time'], columns =['Value'])
            
        np.savetxt(path + 'z_total.csv', z_soln, delimiter=",")
        np.savetxt(path + 'y_total.csv', y_soln, delimiter=",")
        result_df.to_csv(path + 'result_total.csv')  
            
     
    ### Finished all ###
    m.dispose()


    
##############################################################################  
##############################################################################
##############################################################################       



def optimize_rate_fix(scenario, pc, pf, ncp, pv, p, K, closest,
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
    m.Params.MIPGap = MIPGap
    
    total_demand = sum(p)
    if scenario == "current": num_stores = num_current_stores
    if scenario == "total": num_stores = num_total_stores
    
    
    ### Variables ###
    z = m.addVars(num_stores, vtype=GRB.BINARY, name = 'z')
    y = m.addVars(num_tracts * num_stores, lb = 0, ub = 1, name = 'y')
    
    
    ### Objective ###
    m.setObjective(quicksum(pv[k] * y[k] for k in range(num_tracts * num_stores)), gp.GRB.MAXIMIZE)
    
    
    ### Constraints ###
    for j in range(num_stores):
           m.addConstr(quicksum(ncp[i * num_stores + j] * y[i * num_stores + j] for i in range(num_tracts)) <= K * z[j])
        
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
    if scenario == "current":
        store_used = sum(z_soln)
        vaccine_rate = pf @ y_soln / total_demand
        avg_dist = (pc @ y_soln / total_demand) * scale_factor    
        allocation_rate = ncp @ y_soln / total_demand

        result = [store_used, vaccine_rate, avg_dist, allocation_rate, round(end - start,1)]        
        result_df = pd.DataFrame(result, index =['Stores used', 'Vaccination rate', 'Avg distance', 'Allocation rate', 'Time'],\
                                 columns =['Value'])
        
        np.savetxt(path + 'z_current.csv', z_soln, delimiter=",")
        np.savetxt(path + 'y_current.csv', y_soln, delimiter=",")
        result_df.to_csv(path + 'result_current.csv')
    
    elif scenario == "total":
        store_used = sum(z_soln)
        vaccine_rate = (pf @ y_soln / total_demand)
        avg_dist = (pc @ y_soln / total_demand) * scale_factor    
        allocation_rate = ncp @ y_soln / total_demand
                     
        num_current_store_used = sum(z_soln[0 : num_current_stores])
        num_dollar_store_used = store_used - num_current_store_used
        
        dollar_store_demand = ncp[num_current_stores * num_tracts : num_total_stores * num_tracts] @ y_soln[num_current_stores * num_tracts : num_total_stores * num_tracts]
        dollar_store_allocation_rate = dollar_store_demand / total_demand
        
        result = [store_used, vaccine_rate, avg_dist, allocation_rate, num_current_store_used, num_dollar_store_used, dollar_store_allocation_rate, round(end - start,1)]
        result_df = pd.DataFrame(result, index =['Stores used', 'Vaccination rate', 'Avg distance', 'Allocation rate',\
                                                 'Current store used', 'Dollar store used', 'Dollar store allocation rate',\
                                                     'Time'], columns =['Value'])
            

        np.savetxt(path + 'z_total.csv', z_soln, delimiter=",")
        np.savetxt(path + 'y_total.csv', y_soln, delimiter=",")
        result_df.to_csv(path + 'result_total.csv')    
 
    ### Finished all ###
    m.dispose()    
    
    
 
    
 
    
 
    
 
    
 
    