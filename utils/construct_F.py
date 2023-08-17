import pandas as pd
import numpy as np



def import_demand_estimation(Model, capacity, C_total, num_tracts, num_current_stores, Quartile, datadir='/export/storage_covidvaccine/Data/', resultdir='/export/storage_covidvaccine/Result/'):
    
    print("Importing demand estimation...\n")

    if Model in BLP_models: 

        # Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]] # v2
        # F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_BLP(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)

        tract = pd.read_csv(f'{datadir}tract_centroids.csv', delimiter = ",", dtype={'GEOID': int, 'POPULATION': int}) # tract info
        block = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv") # ['zip', 'blkid', 'dist', 'population', 'weights', 'market_ids', 'nodes', 'logdist']
        block.sort_values(by=['blkid'], inplace=True)
        blk_tract = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) # block-tract
        block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{capacity}_200_3q.csv', delimiter = ",") # block estimates
        distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})



        F_D_current, F_D_total, F_DH_current, F_DH_total = construct_F_BLP()



    if Model in LogLin_models: 
        Demand_parameter = [[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, 0.053, 0.047, 0.039]]
        F_D_current, F_D_total, F_DH_current, F_DH_total = construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)

    pass



def construct_F_BLP():

    C_current = C_total[:,0:num_current_stores]
    M_closest_current = np.argpartition(C_current, M, axis=1)[:,:M]
    F_current = np.zeros((num_tracts, num_current_stores))

    for i in range(num_tracts):

        tract_id, tract_pop = tract['GEOID'][i], tract['POPULATION'][i]
        tract_closest_sites = M_closest_current[i]

        blocks_id = blk_tract[blk_tract.tract == tract_id].blkid.to_list()

        for site in tract_closest_sites:

            tract_site_willingess = 0

            for block_id in blocks_id:
                    
                block_pop = block[block.blkid == block_id].population.values[0]
                block_abd, block_distcoef = block_utils[block_utils.blkid == block_id].abd.values[0], block_utils[block_utils.blkid == block_id].distcoef.values[0]
                logdist = distdf[(distdf.blkid == block_id) & (distdf.locid == site)].logdist.values[0]

                block_utility = block_abd + block_distcoef * logdist
                tract_site_willingess += (block_pop / tract_pop) * np.exp(block_utility)/(1 + np.exp(block_utility))

            print(f'New willingness {str(tract_site_willingess)}\n')

            F_current[i][site] = tract_site_willingess


    pass









        






'''
def construct_F_BLP(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile):

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
    
    return F_D_current, F_D_total, F_DH_current, F_DH_total
'''



def construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile):

    F_D_total = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total/1000)
    F_D_current = F_D_total[:,0:num_current_stores]

    F_DH_total = []
    for i in range(num_tracts):
                
        tract_quartile = Quartile[i]
                
        if tract_quartile == 1:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][2]) + (Demand_parameter[1][1] + Demand_parameter[1][5]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 2:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][3]) + (Demand_parameter[1][1] + Demand_parameter[1][6]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 3:
            tract_willingness = (Demand_parameter[1][0] + Demand_parameter[1][4]) + (Demand_parameter[1][1] + Demand_parameter[1][7]) * np.log(C_total[i,:]/1000)
        elif tract_quartile == 4:
            tract_willingness = Demand_parameter[1][0] + Demand_parameter[1][1] * np.log(C_total[i,:]/1000)
        else:
            tract_willingness = Demand_parameter[0][0] + Demand_parameter[0][1] * np.log(C_total[i,:]/1000)
                
        F_DH_total.append(tract_willingness)
                
    F_DH_total = np.asarray(F_DH_total)
    F_DH_current = F_DH_total[:,0:num_current_stores]

    return F_D_current, F_D_total, F_DH_current, F_DH_total




