import pandas as pd
import numpy as np


def import_location(Chain_type="Dollar", M=5, datadir="/export/storage_covidvaccine/Data"):

    
    ### Census Tract 
    Population = np.genfromtxt(f'{datadir}/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    Quartile = np.genfromtxt(f'{datadir}/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)
    
    ### Current ###
    C_current_mat = np.genfromtxt(f'{datadir}/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current_mat = C_current_mat.astype(int)
    C_current_mat = C_current_mat.T
    num_tracts, num_current_stores = np.shape(C_current_mat)

    ### Chains ###
    C_chains_mat = np.genfromtxt(f'{datadir}/CA_dist_matrix_' + Chain_type + '.csv', delimiter = ",", dtype = float)
    C_chains_mat = C_chains_mat.astype(int)
    C_chains_mat = C_chains_mat.T
    num_tracts, num_chains_stores = np.shape(C_chains_mat)
    C_chains_mat = np.where(C_chains_mat < 0, 1317574, C_chains_mat)
    
    ### Total ###
    C_total_mat = np.concatenate((C_current_mat, C_chains_mat), axis = 1)
    num_total_stores = num_current_stores + num_chains_stores
    

    return Population, Quartile, C_current_mat, C_total_mat, num_tracts, num_current_stores, num_total_stores
    


def construct_V(Demand_parameter, C_total, num_tracts, num_current_stores, Quartile):

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



def compute_results(Population, Quartile, y):

    total_population = sum(Population)
    CA_TRACT = pd.DataFrame(data = {'Population': Population, 'HPIQuartile': Quartile})
    
    population1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Population'].values)
    population2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Population'].values)
    population3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Population'].values)
    population4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Population'].values)
    population_vec = [total_population, population1, population2, population3, population4]


    total_vaccinated = np.sum(y, axis = 1)
    CA_TRACT['Vaccinated'] = total_vaccinated

    total_rate = sum(CA_TRACT['Vaccinated'].values) / total_population
    total_rate1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccinated'].values) / population1
    total_rate2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccinated'].values) / population2
    total_rate3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccinated'].values) / population3
    total_rate4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Vaccinated'].values) / population4

    total_rate = np.array([[total_rate, total_rate1, total_rate2, total_rate3, total_rate4]])
    total_vaccination = np.round(total_rate * population_vec / 1000000,2)[0]
    total_proportion = np.round(total_rate * 100)[0]

    CA_TRACT.to_csv('CA_TRACT_DAA.csv', encoding='utf-8', index=False, header=True)

    return total_vaccination, total_proportion