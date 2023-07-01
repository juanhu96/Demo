import os
import numpy as np
os.chdir('/export/storage_covidvaccine/Code')
from utils.optimize_chain import optimize_chain



def optimize_main(Model_list = ['MaxRateHPIDist', 'MaxRateDist', 'MaxRateFixV', 'MinDist'],
                  Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                  K_list = [8000, 10000, 12000],
                  Demand_estimation='BLP'):


    '''
    Parameters
    ----------

    Model_list: List of strings
        List of models to run experiments

    Chain_list: List of strings
        List of partnerships to consider

    K_list: List of int
        List of capacity to consider

    Demand_estimation : string
        Demand function in the optimization objective:
        'BLP', 'linear'

        This is only appliable to 'MaxRateHPIDist' and 'MaxRateDist'
    
    '''   


    datadir = "/export/storage_covidvaccine/Data"
    m1coefs = np.load(f'{datadir}/Analysis/m1coefs.npy')
    m2coefs = np.load(f'{datadir}/Analysis/m2coefs.npy')
    

    if Demand_estimation == 'BLP': Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]]
    # if Demand_estimation == 'BLP': Demand_parameter = [[1.144, -0.567], [1.676, -0.243, -1.101, -0.796, -0.699, -0.331, -0.343, -0.226]]
    if Demand_estimation == 'linear': Demand_parameter = [[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, 0.053, 0.047, 0.039]] # previous result

    print('Coefficients are imported\n')
    print(Demand_parameter)


    for Model in Model_list: 

        model_path = '../Result/' + Model + '/'
        if not os.path.exists(model_path): os.mkdir(model_path)
        
        for K in K_list:

            M = 5
            parameter_path = model_path + 'M' + str(M) + '_K' + str(K) + '/'
            if not os.path.exists(parameter_path): os.mkdir(parameter_path)

            for Chain_type in Chain_list:
                
                chain_path = parameter_path + Chain_type + '/'
                if not os.path.exists(chain_path): os.mkdir(chain_path)

                optimize_chain(Chain_type, Model, M = M, K = K, Demand_estimation = Demand_estimation, Demand_parameter = Demand_parameter, expdirpath = chain_path)
                            
            
            if Model != 'MinDist':
                
                M = 10
                parameter_path = model_path + 'M' + str(M) + '_K' + str(K) + '/'
                if not os.path.exists(parameter_path): os.mkdir(parameter_path)
                
                for Chain_type in Chain_list:
                    
                    chain_path = parameter_path + Chain_type + '/'
                    if not os.path.exists(chain_path): os.mkdir(chain_path)

                    optimize_chain(Chain_type, Model, M = M, K = K, Demand_estimation = Demand_estimation, Demand_parameter = Demand_parameter, expdirpath = chain_path)
    
    
    print('All problems solved!')