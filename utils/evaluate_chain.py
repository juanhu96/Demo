import os
import pandas as pd
import numpy as np
os.chdir('/export/storage_covidvaccine/Code')
from utils.evaluate_model import evaluate_rate
scale_factor = 10000
MAXDIST = 10000000


def evaluate_chain(Chain_type, Model, Demand_parameter, expdirpath, M=5, K=8000):
    
    print('Start evaluating the optimization solution for ' + Chain_type + ' under Model ' + Model + ' with M = '  + str(M) + ' and K = ' + str(K) + '...\n')
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
    
    
    Closest_current = Closest_current.flatten()
    Closest_total = Closest_total.flatten()

    ###########################################################################

    # TODO: do matching between tract & zip, and also include demographics

    Deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
    F_D_total = np.exp(Deltahat) / (1+np.exp(Deltahat))
    F_D_current = F_D_total[:,0:num_current_stores]

    F_DH_total = []
    for i in range(num_tracts):
                
        tract_quartile = Quartile[i]
                
        if tract_quartile == 1:
            deltahat = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        elif tract_quartile == 2:
            deltahat = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        elif tract_quartile == 3:
            deltahat = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        elif tract_quartile == 4:
            deltahat = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
        else:
            deltahat = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
            tract_willingness = np.exp(deltahat) / (1+np.exp(deltahat))
                
        F_DH_total.append(tract_willingness)
                
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
    
    f_dh_current = F_DH_current.flatten()
    f_dh_total = F_DH_total.flatten()
    
    # population * distance 
    pc_current = p_current * c_current
    pc_total = p_total * c_total
    
    # population * willingness
    pfdh_current = p_current * f_dh_current
    pfdh_total = p_total * f_dh_total
    
    
    del C_current, C_total, C_chains, F_D_total, F_D_current, F_DH_total, F_DH_current
    
    ###########################################################################

    # Import optimal z from optimziation

    if Model == 'MinDist': path = '../Result/' + Model + '/' + 'M5_K' + str(K) + '/'  + Chain_type + '/'
    else: path = '../Result/' + Model + '/' + 'M' + str(M) + '_K' + str(K) + '/'  + Chain_type + '/'

    if Model == 'MaxRateHPIDist' or Model == 'MaxRateDist':
        z_total = np.genfromtxt(path + 'z_BLP_total.csv', delimiter = ",", dtype = float)
        z_current = np.genfromtxt(path + 'z_BLP_current.csv', delimiter = ",", dtype = float)

    else:
        z_total = np.genfromtxt(path + 'z_BLP_total.csv', delimiter = ",", dtype = float)
        z_current = np.genfromtxt(path + 'z_BLP_current.csv', delimiter = ",", dtype = float)

    ###########################################################################


    if Chain_type == 'Dollar':
        evaluate_rate(scenario = 'current', z = z_current,
                      pc = pc_current, pf = pfdh_current, ncp = p_current, p = Population,
                      closest = Closest_current, K=K, 
                      num_current_stores=num_current_stores,
                      num_total_stores=num_total_stores, 
                      num_tracts=num_tracts,
                      scale_factor=scale_factor,
                      path = expdirpath)

    evaluate_rate(scenario = 'total', z = z_total,
                  pc = pc_total, pf = pfdh_total, ncp = p_total, p = Population, 
                  closest = Closest_total, K=K,
                  num_current_stores=num_current_stores,
                  num_total_stores=num_total_stores,
                  num_tracts=num_tracts,
                  scale_factor=scale_factor,
                  path = expdirpath)

    # expdirpath already has the Model info so we don't need Model