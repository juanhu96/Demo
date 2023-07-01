import os
import pandas as pd
import numpy as np
os.chdir('/export/storage_covidvaccine/Code')
from utils.optimize_model import optimize_rate, optimize_dist, optimize_rate_fix
scale_factor = 10000
MAXDIST = 10000000


def optimize_chain(Chain_type, Model, Demand_parameter, expdirpath, M=5, K=8000, Demand_estimation='BLP'):
    
    print('Start importing the problem for ' + Chain_type + ' under Model ' + Model + ' with ' + Demand_estimation + ' demand function, M = '  + str(M) + ' and K = ' + str(K) + '...\n')
    print(expdirpath)
    
    ###########################################################################
    
    ### Census Tract 
    Population = np.genfromtxt('../Data/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    Quartile = np.genfromtxt('../Data/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)
    
    ### Current ###
    C_current = np.genfromtxt('../Data/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current = C_current.astype(int)
    C_current = C_current.T
    num_tracts, num_current_stores = np.shape(C_current)

    ### Chains ###
    C_chains = np.genfromtxt('../Data/CA_dist_matrix_' + Chain_type + '.csv', delimiter = ",", dtype = float)
    C_chains = C_chains.astype(int)
    C_chains = C_chains.T
    num_tracts, num_chains_stores = np.shape(C_chains)
    ## Avoid negative numbers for high schools
    C_chains = np.where(C_chains < 0, 1317574, C_chains)
    
    ### Total ###
    C_total = np.concatenate((C_current, C_chains), axis = 1)
    num_total_stores = num_current_stores + num_chains_stores
    
    ###########################################################################
    
    ### Travel to the closest M stores only
    Closest_current = np.ones((num_tracts, num_current_stores))
    Closest_total = np.ones((num_tracts, num_total_stores))
    np.put_along_axis(Closest_current, np.argpartition(C_current,M,axis=1)[:,M:],0,axis=1)
    np.put_along_axis(Closest_total, np.argpartition(C_total,M,axis=1)[:,M:],0,axis=1)
    
    ### Newly added chains does not affect current rank
    Closest_total_chains = np.ones((num_tracts, num_chains_stores))
    np.put_along_axis(Closest_total_chains, np.argpartition(C_chains,M,axis=1)[:,M:],0,axis=1)    
    Closest_total_chains = np.concatenate((Closest_current, Closest_total_chains), axis = 1)
    
    ## Variation of MinDist
    C_current_new = C_current * Closest_current
    C_total_new = C_total * Closest_total_chains
    C_current_new = np.where(C_current_new == 0, MAXDIST, C_current_new)
    C_total_new = np.where(C_total_new == 0, MAXDIST, C_total_new)
    

    Closest_current = Closest_current.flatten()
    Closest_total = Closest_total.flatten()
    Closest_total_chains = Closest_total_chains.flatten()
            
    ###########################################################################

    if Demand_estimation == 'BLP':

        Deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
        F_D_total = np.exp(Deltahat) / (1+np.exp(Deltahat))
        F_D_current = F_D_total[:,0:num_current_stores]

        F_DH_total = []
        for i in range(num_tracts):
                
            zip_quartile = Quartile[i]
                
            if zip_quartile == 1:
                deltahat = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
                zip_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
            elif zip_quartile == 2:
                deltahat = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
                zip_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
            elif zip_quartile == 3:
                deltahat = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
                zip_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
            elif zip_quartile == 4:
                deltahat = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
                zip_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
            else:
                deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
                zip_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                
            F_DH_total.append(zip_willingness)
                
        F_DH_total = np.asarray(F_DH_total)
        F_DH_current = F_DH_total[:,0:num_current_stores]
    

    elif Demand_estimation == 'linear':

        F_D_total = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
        F_D_current = F_D_total[:,0:num_current_stores]

        F_DH_total = []
        for i in range(num_tracts):
                
            zip_quartile = Quartile[i]
                
            if zip_quartile == 1:
                zip_willingness = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
            elif zip_quartile == 2:
                zip_willingness = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
            elif zip_quartile == 3:
                zip_willingness = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
            elif zip_quartile == 4:
                zip_willingness = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
            else:
                zip_willingness = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
                
            F_DH_total.append(zip_willingness)
                
        F_DH_total = np.asarray(F_DH_total)
        F_DH_current = F_DH_total[:,0:num_current_stores]
    

    ###########################################################################
    
    # n copies of demand
    p_total = np.tile(Population, num_total_stores)
    p_total = np.reshape(p_total, (num_total_stores, num_tracts))
    p_total = p_total.T.flatten()
       
    p_current = np.tile(Population, num_current_stores)
    p_current = np.reshape(p_current, (num_current_stores, num_tracts))
    p_current = p_current.T.flatten()
    
    # travel distance
    c_current = C_current.flatten() / scale_factor
    c_total = C_total.flatten() / scale_factor
    
    f_d_current = F_D_current.flatten()
    f_d_total = F_D_total.flatten()
    
    f_dh_current = F_DH_current.flatten()
    f_dh_total = F_DH_total.flatten()
    
    # population * distance 
    pc_current = p_current * c_current
    pc_total = p_total * c_total
    
    # population * willingness
    pfd_current = p_current * f_d_current
    pfd_total = p_total * f_d_total
    
    pfdh_current = p_current * f_dh_current
    pfdh_total = p_total * f_dh_total
    
    # population * fix willingness/vaccination rate
    fix_vac_rate = 0.7 # vaccination rate as of March, 2022
    pv_current = p_current * fix_vac_rate
    pv_total = p_total * fix_vac_rate
    
    del C_current, C_total, C_chains, F_D_total, F_D_current, F_DH_total, F_DH_current
    
    ###########################################################################

    if Model == 'MaxRateHPIDist':

        if Chain_type == 'Dollar':
            optimize_rate(scenario = 'current', Demand_estimation = Demand_estimation,
                          pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                          closest = Closest_current, K=K, 
                          num_current_stores=num_current_stores,
                          num_total_stores=num_total_stores, 
                          num_tracts=num_tracts,
                          scale_factor=scale_factor,
                          path = expdirpath)
             
        optimize_rate(scenario = 'total', Demand_estimation = Demand_estimation,
                   pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                   closest = Closest_total_chains, K=K,
                   num_current_stores=num_current_stores,
                   num_total_stores=num_total_stores,
                   num_tracts=num_tracts,
                   scale_factor=scale_factor,
                   path = expdirpath)

    ###########################################################################
    
    if Model == 'MaxRateDist':
        
        if Chain_type == 'Dollar':
            optimize_rate(scenario = 'current', Demand_estimation = Demand_estimation,
                          pc = pc_current, pf = pfd_current, ncp = p_current, p = Population,
                          closest = Closest_current, K=K, 
                          num_current_stores=num_current_stores, 
                          num_total_stores=num_total_stores, 
                          num_tracts=num_tracts,
                          scale_factor=scale_factor, 
                          path = expdirpath)
            
        optimize_rate(scenario = 'total', Demand_estimation = Demand_estimation,
                      pc = pc_total, pf = pfd_total, ncp = p_total, p = Population,
                      closest = Closest_total_chains, K=K,
                      num_current_stores=num_current_stores, 
                      num_total_stores=num_total_stores,
                      num_tracts=num_tracts,
                      scale_factor=scale_factor,
                      path = expdirpath)
  
    ###########################################################################
    
    if Model == 'MinDist':
        
        if Chain_type == 'Dollar':
            optimize_dist(scenario = 'current', 
                          pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population, K=K, 
                          num_current_stores=num_current_stores,
                          num_total_stores=num_total_stores, 
                          num_tracts=num_tracts, 
                          scale_factor=scale_factor,
                          path = expdirpath)
       
        optimize_dist(scenario = 'total', 
                      pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, K=K,
                      num_current_stores=num_current_stores,
                      num_total_stores=num_total_stores, 
                      num_tracts=num_tracts, 
                      scale_factor=scale_factor, 
                      path = expdirpath)  



    ###########################################################################

    if Model == 'MinDistNew':

        c_current_new = C_current_new.flatten() / scale_factor
        c_total_new = C_total_new.flatten() / scale_factor

        pc_current_new = p_current * c_current_new
        pc_total_new = p_total * c_total_new

        
        if Chain_type == 'Dollar':
            optimize_dist(scenario = 'current',
                              pc = pc_current_new, pf = pfdh_current, ncp = p_current, p = Population, K=K, 
                              num_current_stores=num_current_stores,
                              num_total_stores=num_total_stores, 
                              num_tracts=num_tracts, 
                              scale_factor=scale_factor,
                              path = expdirpath)
       
        optimize_dist(scenario = 'total',
                          pc = pc_total_new, pf = pfdh_total, ncp = p_total, p = Population, K=K,
                          num_current_stores=num_current_stores,
                          num_total_stores=num_total_stores, 
                          num_tracts=num_tracts, 
                          scale_factor=scale_factor, 
                          path = expdirpath)   
        
    ###########################################################################
    
    if Model == 'MaxRateFixV':
            
        if Chain_type == 'Dollar':
            optimize_rate_fix(scenario = 'current', chain_type = Chain_type,
                               pc = pc_current, pf = pfdh_current, ncp = p_current, pv = pv_current, p = Population,
                               closest = Closest_current, K=K, 
                               num_current_stores=num_current_stores,
                               num_total_stores=num_total_stores,
                               num_tracts=num_tracts, 
                               scale_factor=scale_factor,
                               path = expdirpath)

        optimize_rate_fix(scenario = 'total', chain_type = Chain_type,
                           pc = pc_total, pf = pfdh_total, ncp = p_total, pv = pv_total, p = Population, 
                           closest = Closest_total_chains, K=K, 
                           num_current_stores=num_current_stores,
                           num_total_stores=num_total_stores, 
                           num_tracts=num_tracts, 
                           scale_factor=scale_factor, 
                           path = expdirpath)  
                