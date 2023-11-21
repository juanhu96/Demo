def partnerships_summary_old(Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist'],
                        Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries'],
                        M_list = [5, 10], K_list = [8000, 10000, 12000],
                        Vaccination_estimation = 'BLP', constraint_list = ['assigned', 'vaccinated'], 
                        export_tract_table = False, filename = ''):
    


    '''
    Summary for each (Model, Chain, M, K, opt_constr, eval_constr) pair

    Parameters
    ----------
    Model_list: List of strings
        List of models to construct summary table

    Chain_list: List of strings
        List of partnerships to construct summary table

    K_list: List of int
        List of capacity to construct summary table
        For 'MinDist' this is only feasible for K = 10000, 12000

    Vaccination_estimation : string
        Demand function used for computing predicted vaccination
        'BLP', 'linear'

    export_tract_table: Boolean
        Whether to export a tract-level summary table (for diagonsis)

    filename : string
        Filename
    '''


    
    chain_summary_table = []


    ## Demand coefficients
    # Demand_parameter = [[1.227, -0.452], [2.028, -0.120, -1.357, -1.197, -0.925, -0.254, -0.218, -0.114]] # v3
    if Vaccination_estimation == 'BLP': Demand_parameter = [[1.227, -0.452], [1.729, -0.031, -0.998, -0.699, -0.614, -0.363, -0.363, -0.249]] # v2
    if Vaccination_estimation == 'linear': Demand_parameter = [[0.755, -0.069], [0.826, -0.016, -0.146, -0.097, -0.077, 0.053, 0.047, 0.039]]
    

    ## Population and quartiles for CA
    Quartile = np.genfromtxt(f'{maindir}/Data/HPIQuartile_TRACT.csv', delimiter = ",", dtype = int)    
    Population = np.genfromtxt(f'{maindir}/Data/CA_demand_over_5.csv', delimiter = ",", dtype = int)
    

    ## Distance matrix for pharmacies (current)
    C_current = np.genfromtxt(f'{maindir}/Data/CA_dist_matrix_current.csv', delimiter = ",", dtype = float)
    C_current = C_current.astype(int)
    C_current = C_current.T
    num_tracts, num_current_stores = np.shape(C_current)
   

    for Model in Model_list:
        for Chain_type in Chain_list:


            ## Distance matrix for chain
            C_chains = np.genfromtxt(f'{maindir}/Data/CA_dist_matrix_{Chain_type}.csv', delimiter = ",", dtype = float)
            C_chains = C_chains.astype(int)
            C_chains = C_chains.T
            num_tracts, num_chain_stores = np.shape(C_chains)
            C_chains = np.where(C_chains < 0, 1317574, C_chains) # High schools
            
            C_total = np.concatenate((C_current, C_chains), axis = 1)
            num_total_stores = num_current_stores + num_chain_stores
            
            C_current_walkable = np.where(C_current < 1600, 1, 0)
            C_chains_walkable = np.where(C_chains < 1600, 1, 0)
            C_total_walkable = np.where(C_total < 1600, 1, 0)
            

            ## Demand matrix for chain
            if Vaccination_estimation == 'BLP': 
            
                ## TODO: also K = 10000 is temp, now F is K depend and need to be imported inside the loop
                F_D_current, F_D_total, F_DH_current, F_DH_total = import_BLP_estimation(Chain_type, 10000)

            if Vaccination_estimation == 'linear': F_D_current, F_D_total, F_DH_current, F_DH_total  = construct_F_LogLin(Model, Demand_parameter, C_total, num_tracts, num_current_stores, Quartile)

            
            for M in M_list:

                # M closest stores only
                Closest_current = np.ones((num_tracts, num_current_stores))
                Closest_total = np.ones((num_tracts, num_total_stores))
                np.put_along_axis(Closest_current, np.argpartition(C_current,M,axis=1)[:,M:],0,axis=1)
                np.put_along_axis(Closest_total, np.argpartition(C_total,M,axis=1)[:,M:],0,axis=1)

                Farthest_current = np.ones((num_tracts, num_current_stores)) - Closest_current
                Farthest_total= np.ones((num_tracts, num_total_stores)) - Closest_total

                Closest_current = Closest_current.flatten()
                Closest_total = Closest_total.flatten()

                Farthest_current = Farthest_current.flatten()
                Farthest_total = Farthest_total.flatten()

                ###########################################################################

                for K in K_list:  

                    if Model in ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV']:

                        for opt_constr in constraint_list:

                            path = f'{maindir}/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/{opt_constr}/'
                            
                            eval_constr = opt_constr

                            # for eval_constr in constraint_list:

                            mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("total", path, eval_constr, num_tracts, num_current_stores, num_total_stores, Closest_total, Farthest_total)

                            chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, eval_constr,\
                            Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_total, C_total, C_total_walkable)

                            chain_summary_table.append(chain_summary)


                            if Chain_type == 'Dollar':

                                mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("current", path, eval_constr, num_tracts, num_current_stores, num_total_stores, Closest_current, Farthest_current)

                                chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, opt_constr, eval_constr,\
                                Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_current, C_current, C_current_walkable)

                                chain_summary_table.append(chain_summary)

                                    # if export_tract_table==True:

                                    #     tract_summary(Model, Chain_type, M, K, Quartile, Population,\
                                    #     mat_y_current_hpi, mat_y_current_hpi_closest, mat_y_current_hpi_farthest, result_current_hpi, F_DH_current, C_current, C_current_walkable,\
                                    #     mat_y_total_hpi, mat_y_total_hpi_closest, mat_y_total_hpi_farthest, result_total_hpi, F_DH_total, C_total, C_total_walkable)



                            ###########################################################################
                            # first round


                            mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("firstround_total", path, "none", num_tracts, num_current_stores, num_total_stores, Closest_total, Farthest_total)

                            chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, opt_constr, "none",\
                            Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_total, C_total, C_total_walkable)

                            chain_summary_table.append(chain_summary)


                            if Chain_type == 'Dollar':

                                mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("firstround_current", path, "none", num_tracts, num_current_stores, num_total_stores, Closest_current, Farthest_current)

                                chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, opt_constr, "none",\
                                Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_current, C_current, C_current_walkable)

                                chain_summary_table.append(chain_summary)



                    else: # MinDist, no opt_constr

                        path = f'{maindir}/Result/{Model}/M{str(M)}_K{str(K)}/{Chain_type}/'

                        for eval_constr in constraint_list:

                            mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("total", path, eval_constr, num_tracts, num_current_stores, num_total_stores, Closest_total, Farthest_total)

                            chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, "none", eval_constr,\
                            Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_total, C_total, C_total_walkable)

                            chain_summary_table.append(chain_summary)


                            if Chain_type == 'Dollar':

                                mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("current", path, eval_constr, num_tracts, num_current_stores, num_total_stores, Closest_current, Farthest_current)

                                chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, "none", eval_constr,\
                                Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_current, C_current, C_current_walkable)

                                chain_summary_table.append(chain_summary)
                        


                        mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("firstround_total", path, "none", num_tracts, num_current_stores, num_total_stores, Closest_total, Farthest_total)

                        chain_summary = create_row('Pharmacy + ' + Chain_type, Model, Chain_type, M, K, "none", "none",\
                        Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_total, C_total, C_total_walkable)

                        chain_summary_table.append(chain_summary)


                        if Chain_type == 'Dollar':

                            mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R = import_solution("firstround_current", path, "none", num_tracts, num_current_stores, num_total_stores, Closest_current, Farthest_current)

                            chain_summary = create_row('Pharmacy-only', Model, Chain_type, M, K, "none", "none",\
                            Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH_current, C_current, C_current_walkable)

                            chain_summary_table.append(chain_summary)


                        
    chain_summary = pd.DataFrame(chain_summary_table)
    chain_summary.to_csv(f'{maindir}Result/sensitivity_results_{filename}_old.csv', encoding='utf-8', index=False, header=True)





def import_solution(scenario, path, eval_constr, num_tracts, num_current_stores, num_total_stores, Closest, Farthest):


    if scenario == "total":
        y_hpi = np.genfromtxt(f'{path}y_total_eval_{eval_constr}.csv', delimiter = ",", dtype = float)
        z_hpi = np.genfromtxt(f'{path}z_total.csv', delimiter = ",", dtype = float)
        num_stores = num_total_stores
        R = num_current_stores - sum(z_hpi[0 : num_current_stores])
    
    elif scenario == "firstround_total":
        y_hpi = np.genfromtxt(f'{path}y_total.csv', delimiter = ",", dtype = float)
        z_hpi = np.genfromtxt(f'{path}z_total.csv', delimiter = ",", dtype = float)
        num_stores = num_total_stores
        R = num_current_stores - sum(z_hpi[0 : num_current_stores])
    
    elif scenario == "current":
        y_hpi = np.genfromtxt(f'{path}y_current_eval_{eval_constr}.csv', delimiter = ",", dtype = float)
        num_stores = num_current_stores
        R = 0

    elif scenario == "firstround_current":
        y_hpi = np.genfromtxt(f'{path}y_current.csv', delimiter = ",", dtype = float)
        num_stores = num_current_stores
        R = 0
    
    else:
        print("Warning: scenario undefined in import_solution().")

    y_hpi_closest = y_hpi * Closest
    y_hpi_farthest = y_hpi * Farthest

    mat_y_hpi = np.reshape(y_hpi, (num_tracts, num_stores))
    mat_y_hpi_closest = np.reshape(y_hpi_closest, (num_tracts, num_stores))
    mat_y_hpi_farthest = np.reshape(y_hpi_farthest, (num_tracts, num_stores))

    return mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R




def create_row(Scenario, Model, Chain_type, M, K, opt_constr, eval_constr, Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH, C, C_walkable):

    
    total_population = sum(Population)
    CA_TRACT = pd.DataFrame(data = {'Population': Population, 'HPIQuartile': Quartile})
    
    population1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Population'].values)
    population2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Population'].values)
    population3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Population'].values)
    population4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Population'].values)
    population_vec = [total_population, population1, population2, population3, population4]

    
    ########################################################################### 
    
    ## Assigned within M
    assigned_within = np.sum(mat_y_hpi_closest, axis = 1)
    CA_TRACT['Assigned_Rate_WithinM'] = assigned_within
    CA_TRACT['Assigned_Population_WithinM'] = CA_TRACT['Assigned_Rate_WithinM'] * CA_TRACT['Population']
    assigned_within_hpi = sum(CA_TRACT['Assigned_Population_WithinM'].values) / total_population
    assigned_within_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_WithinM'].values) / population1
    assigned_within_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_WithinM'].values) / population2
    assigned_within_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_WithinM'].values) / population3
    assigned_within_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Assigned_Population_WithinM'].values) / population4
    
    assigned_within_rate = np.array([[assigned_within_hpi, assigned_within_hpi1, assigned_within_hpi2, assigned_within_hpi3, assigned_within_hpi4]])
    assigned_within_population = np.round(assigned_within_rate * population_vec / 1000000,4)[0]
    assigned_within_rate = np.round(assigned_within_rate * 100)[0]


    ## Assigned away M
    assigned_away = np.sum(mat_y_hpi_farthest, axis = 1)
    CA_TRACT['Assigned_Rate_AwayM'] = assigned_away
    CA_TRACT['Assigned_Population_AwayM'] = CA_TRACT['Assigned_Rate_AwayM'] * CA_TRACT['Population']
    assigned_away_hpi = sum(CA_TRACT['Assigned_Population_AwayM'].values) / total_population
    assigned_away_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_AwayM'].values) / population1
    assigned_away_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_AwayM'].values) / population2
    assigned_away_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_AwayM'].values) / population3
    assigned_away_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Assigned_Population_AwayM'].values) / population4
    
    assigned_away_rate = np.array([[assigned_away_hpi, assigned_away_hpi1, assigned_away_hpi2, assigned_away_hpi3, assigned_away_hpi4]])
    assigned_away_population = np.round(assigned_away_rate * population_vec / 1000000,4)[0]
    assigned_away_rate = np.round(assigned_away_rate * 100)[0]


    ## Total vaccination
    total_rate_hpi = np.sum(np.multiply(F_DH, mat_y_hpi), axis = 1)
    CA_TRACT['Rate_HPI'] = total_rate_hpi
    CA_TRACT['Vaccinated_Population_HPI'] = CA_TRACT['Rate_HPI'] * CA_TRACT['Population']
    
    total_rate_hpi = sum(CA_TRACT['Vaccinated_Population_HPI'].values) / total_population
    total_rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccinated_Population_HPI'].values) / population1
    total_rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccinated_Population_HPI'].values) / population2
    total_rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccinated_Population_HPI'].values) / population3
    total_rate_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Vaccinated_Population_HPI'].values) / population4

    total_rate = np.array([[total_rate_hpi, total_rate_hpi1, total_rate_hpi2, total_rate_hpi3, total_rate_hpi4]])
    total_vaccination = np.round(total_rate * population_vec / 1000000,4)[0]
    total_proportion = np.round(total_rate * 100)[0]


    ## Vaccination within M
    rate_hpi = np.sum(np.multiply(F_DH, mat_y_hpi_closest), axis = 1)
    CA_TRACT['Rate_HPI'] = rate_hpi
    CA_TRACT['Vaccinated_Population_HPI'] = CA_TRACT['Rate_HPI'] * CA_TRACT['Population']
    
    rate_hpi = sum(CA_TRACT['Vaccinated_Population_HPI'].values) / total_population
    rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccinated_Population_HPI'].values) / population1
    rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccinated_Population_HPI'].values) / population2
    rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccinated_Population_HPI'].values) / population3
    rate_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Vaccinated_Population_HPI'].values) / population4
    
    rate = np.array([[rate_hpi, rate_hpi1, rate_hpi2, rate_hpi3, rate_hpi4]])
    vaccination = np.round(rate * population_vec / 1000000,4)[0]
    proportion = np.round(rate * 100)[0]
    

    ## Vaccination away M
    drop_hpi = np.sum(np.multiply(F_DH, mat_y_hpi_farthest), axis = 1)
    CA_TRACT['Drop_HPI'] = drop_hpi
    CA_TRACT['Drop_Vaccination_HPI'] = CA_TRACT['Drop_HPI'] * CA_TRACT['Population']
    
    drop_hpi = sum(CA_TRACT['Drop_Vaccination_HPI'].values) / total_population
    drop_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Drop_Vaccination_HPI'].values) / population1
    drop_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Drop_Vaccination_HPI'].values) / population2
    drop_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Drop_Vaccination_HPI'].values) / population3
    drop_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Drop_Vaccination_HPI'].values) / population4
    
    drop = np.array([[drop_hpi, drop_hpi1, drop_hpi2, drop_hpi3, drop_hpi4]])
    drop_vaccination = np.round(drop * population_vec / 1000000,4)[0]
    drop_proportion = np.round(drop * 100)[0]
    
    
    ## Assigned within Walkable
    assigned_walkable_hpi = np.sum(np.multiply(C_walkable, mat_y_hpi), axis=1) 
    CA_TRACT['Assigned_Walkable_HPI'] = assigned_walkable_hpi
    CA_TRACT['Assigned_Walkable_Population_HPI'] = CA_TRACT['Assigned_Walkable_HPI'] * CA_TRACT['Population']    

    assigned_walkable_hpi = sum(CA_TRACT['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Assigned_Walkable_Population_HPI'].values)

    assigned_walkable_list = np.array([[assigned_walkable_hpi, assigned_walkable_hpi1, assigned_walkable_hpi2, assigned_walkable_hpi3, assigned_walkable_hpi4]])
    assigned_walkable = np.round(assigned_walkable_list / 1000000, 4)[0]
    assigned_walkable_proportion = np.round(assigned_walkable_list[0] / population_vec * 100)


    ## Vaccination within Walkable
    rate_walkable_hpi = np.sum(np.multiply(C_walkable, np.multiply(F_DH, mat_y_hpi)), axis =1) 
    CA_TRACT['Vaccinate_Walkable_HPI'] = rate_walkable_hpi
    CA_TRACT['Vaccination_Walkable_HPI'] = CA_TRACT['Vaccinate_Walkable_HPI'] * CA_TRACT['Population']    

    rate_walkable_hpi = sum(CA_TRACT['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Vaccination_Walkable_HPI'].values)
    
    rate_walkable_list = np.array([[rate_walkable_hpi, rate_walkable_hpi1, rate_walkable_hpi2, rate_walkable_hpi3, rate_walkable_hpi4]])
    vaccination_walkable = np.round(rate_walkable_list / 1000000, 4)[0]
    rate_walkable = np.round(rate_walkable_list[0] / population_vec * 100)


    ## Distance among assigned
    assigned_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi), axis = 1) / np.sum(mat_y_hpi, axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Assigned'] = assigned_dist_hpi
    CA_TRACT['Assigned_Population_HPI'] = CA_TRACT['Population'] * np.sum(mat_y_hpi, axis = 1)
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Assigned'] * CA_TRACT['Assigned_Population_HPI']
    
    assigned_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Assigned_Population_HPI'].values)
    assigned_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_HPI'])
    assigned_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_HPI'])
    assigned_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_HPI'])
    assigned_dist_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Assigned_Population_HPI'])
    
    assigned_dist_actual = np.array([[assigned_dist_hpi, assigned_dist_hpi1, assigned_dist_hpi2, assigned_dist_hpi3, assigned_dist_hpi4]])
    assigned_dist_actual = np.round(assigned_dist_actual/1000, 4)[0]
    

    ## Distance among assigned within M
    closest_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi_closest), axis = 1) / np.sum(mat_y_hpi_closest, axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Closest'] = closest_dist_hpi
    CA_TRACT['Closest_Population_HPI'] = np.sum(mat_y_hpi_closest, axis = 1) * CA_TRACT['Population']
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Closest'] * CA_TRACT['Closest_Population_HPI']

    closest_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Closest_Population_HPI'].values)
    closest_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Closest_Population_HPI'])
    closest_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Closest_Population_HPI'])
    closest_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Closest_Population_HPI'])
    closest_dist_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Closest_Population_HPI'])

    closest_dist_actual = np.array([[closest_dist_hpi, closest_dist_hpi1, closest_dist_hpi2, closest_dist_hpi3, closest_dist_hpi4]])
    closest_dist_actual = np.round(closest_dist_actual/1000, 4)[0]
    

    ## Distance among assigned away M
    farthest_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi_farthest), axis = 1) / np.sum(mat_y_hpi_farthest, axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Farthest'] = farthest_dist_hpi
    CA_TRACT['Farthest_Population_HPI'] = np.sum(mat_y_hpi_farthest, axis = 1) * CA_TRACT['Population']
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Farthest'] * CA_TRACT['Farthest_Population_HPI']

    farthest_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Farthest_Population_HPI'].values)
    farthest_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Farthest_Population_HPI'])
    farthest_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Farthest_Population_HPI'])
    farthest_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Farthest_Population_HPI'])
    farthest_dist_hpi4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Farthest_Population_HPI'])

    farthest_dist_actual = np.array([[farthest_dist_hpi, farthest_dist_hpi1, farthest_dist_hpi2, farthest_dist_hpi3, farthest_dist_hpi4]])
    farthest_dist_actual = np.round(farthest_dist_actual/1000, 4)[0]
    

    ###########################################################################

    chain_summary = {'Model': Model, 'Chain': Scenario,
                     'Opt Constr': opt_constr, 'Eval Constr': eval_constr,
                     'M': M, 'K': K,
                     'R': R,
                     'Vaccination': total_vaccination[0], 
                     'Vaccination HPI1': total_vaccination[1], 'Vaccination HPI2': total_vaccination[2], 
                     'Vaccination HPI3': total_vaccination[3], 'Vaccination HPI4': total_vaccination[4],
                     'Rate': total_proportion[0],
                     'Rate HPI1': total_proportion[1], 'Rate HPI2': total_proportion[2], 
                     'Rate HPI3': total_proportion[3], 'Rate HPI4': total_proportion[4],
                     'Vaccination Within M': vaccination[0], 
                     'Vaccination Within M HPI1': vaccination[1], 'Vaccination Within M HPI2': vaccination[2], 
                     'Vaccination Within M HPI3': vaccination[3], 'Vaccination Within M HPI4': vaccination[4],
                     'Rate Within M': proportion[0],
                     'Rate Within M HPI1': proportion[1], 'Rate Within M HPI2': proportion[2], 
                     'Rate Within M HPI3': proportion[3], 'Rate Within M HPI4': proportion[4],
                    #  'Vaccination Away M': drop_vaccination[0], 
                    #  'Vaccination Away M HPI1': drop_vaccination[1], 'Vaccination Away M HPI2': drop_vaccination[2], 
                    #  'Vaccination Away M HPI3': drop_vaccination[3], 'Vaccination Away M HPI4': drop_vaccination[4],
                    #  'Rate Away M': drop_proportion[0],
                    #  'Rate Away M HPI1': drop_proportion[1], 'Rate Away M HPI2': drop_proportion[2], 
                    #  'Rate Away M HPI3': drop_proportion[3], 'Rate Away M HPI4': drop_proportion[4],
                     'Assigned Within M': assigned_within_population[0], 
                     'Assigned Within M HPI1': assigned_within_population[1], 'Assigned Within M HPI2': assigned_within_population[2], 
                     'Assigned Within M HPI3': assigned_within_population[3], 'Assigned Within M HPI4': assigned_within_population[4],
                     'Assigned Rate Within M': assigned_within_rate[0], 
                     'Assigned Rate Within M HPI1': assigned_within_rate[1], 'Assigned Rate Within M HPI2': assigned_within_rate[2], 
                     'Assigned Rate Within M HPI3': assigned_within_rate[3], 'Assigned Rate Within M HPI4': assigned_within_rate[4],
                    #  'Assigned Away M': assigned_away_population[0], 
                    #  'Assigned Away M HPI1': assigned_away_population[1], 'Assigned Away M HPI2': assigned_away_population[2], 
                    #  'Assigned Away M HPI3': assigned_away_population[3], 'Assigned Away M HPI4': assigned_away_population[4],
                    #  'Assigned Rate Away M': assigned_away_rate[0], 
                    #  'Assigned Rate Away M HPI1': assigned_away_rate[1], 'Assigned Rate Away M HPI2': assigned_away_rate[2], 
                    #  'Assigned Rate Away M HPI3': assigned_away_rate[3], 'Assigned Rate Away M HPI4': assigned_away_rate[4],
                     'Assigned Walkable': assigned_walkable[0], 
                     'Assigned Walkable HPI1': assigned_walkable[1], 'Assigned Walkable HPI2': assigned_walkable[2], 
                     'Assigned Walkable HPI3': assigned_walkable[3], 'Assigned Walkable HPI4': assigned_walkable[4],
                     'Assigned Walkable rate': assigned_walkable_proportion[0], 
                     'Assigned Walkable rate HPI1': assigned_walkable_proportion[1], 'Assigned Walkable rate HPI2': assigned_walkable_proportion[2],
                     'Assigned Walkable rate HPI3': assigned_walkable_proportion[3], 'Assigned Walkable rate HPI4': assigned_walkable_proportion[4],
                     'Vaccination Walkable': vaccination_walkable[0], 
                     'Vaccination Walkable HPI1': vaccination_walkable[1], 'Vaccination Walkable HPI2': vaccination_walkable[2], 
                     'Vaccination Walkable HPI3': vaccination_walkable[3], 'Vaccination Walkable HPI4': vaccination_walkable[4],
                     'Vaccination Walkable rate': rate_walkable[0], 
                     'Vaccination Walkable rate HPI1': rate_walkable[1], 'Vaccination Walkable rate HPI2': rate_walkable[2],
                     'Vaccination Walkable rate HPI3': rate_walkable[3], 'Vaccination Walkable rate HPI4': rate_walkable[4],
                     'Average distance': assigned_dist_actual[0],
                     'Average distance HPI1': assigned_dist_actual[1], 'Average distance HPI2': assigned_dist_actual[2],
                     'Average distance HPI3': assigned_dist_actual[3], 'Average distance HPI4': assigned_dist_actual[4]
                    #  'Closest distance': closest_dist_actual[0],
                    #  'Closest distance HPI1': closest_dist_actual[1], 'Closest distance HPI2': closest_dist_actual[2],
                    #  'Closest distance HPI3': closest_dist_actual[3], 'Closest distance HPI4': closest_dist_actual[4],
                    #  'Farthest distance': farthest_dist_actual[0],
                    #  'Farthest distance HPI1': farthest_dist_actual[1], 'Farthest distance HPI2': farthest_dist_actual[2],
                    #  'Farthest distance HPI3': farthest_dist_actual[3], 'Farthest distance HPI4': farthest_dist_actual[4]
                    }                   



    return chain_summary




def create_row(Scenario, Model, Chain_type, M, K, opt_constr, eval_constr, Quartile, Population, mat_y_hpi, mat_y_hpi_closest, mat_y_hpi_farthest, R, F_DH, C, C_walkable):

    
    maindir = '/export/storage_covidvaccine/'
    Quartile = np.genfromtxt(f'{maindir}/Data/HPIQuantile3_TRACT.csv', delimiter = ",", dtype = int)  

    total_population = sum(Population)
    CA_TRACT = pd.DataFrame(data = {'Population': Population, 'HPIQuartile': Quartile})
    
    population1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Population'].values)
    population2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Population'].values)
    population3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Population'].values)
    population_vec = [total_population, population1, population2, population3]

    
    ########################################################################### 
    
    ## Assigned within M
    assigned_within = np.sum(mat_y_hpi_closest, axis = 1)
    CA_TRACT['Assigned_Rate_WithinM'] = assigned_within
    CA_TRACT['Assigned_Population_WithinM'] = CA_TRACT['Assigned_Rate_WithinM'] * CA_TRACT['Population']
    assigned_within_hpi = sum(CA_TRACT['Assigned_Population_WithinM'].values) / total_population
    assigned_within_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_WithinM'].values) / population1
    assigned_within_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_WithinM'].values) / population2
    assigned_within_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_WithinM'].values) / population3
    
    assigned_within_rate = np.array([[assigned_within_hpi, assigned_within_hpi1, assigned_within_hpi2, assigned_within_hpi3]])
    assigned_within_population = np.round(assigned_within_rate * population_vec / 1000000,4)[0]
    assigned_within_rate = np.round(assigned_within_rate * 100)[0]


    ## Assigned away M
    assigned_away = np.sum(mat_y_hpi_farthest, axis = 1)
    CA_TRACT['Assigned_Rate_AwayM'] = assigned_away
    CA_TRACT['Assigned_Population_AwayM'] = CA_TRACT['Assigned_Rate_AwayM'] * CA_TRACT['Population']
    assigned_away_hpi = sum(CA_TRACT['Assigned_Population_AwayM'].values) / total_population
    assigned_away_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_AwayM'].values) / population1
    assigned_away_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_AwayM'].values) / population2
    assigned_away_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_AwayM'].values) / population3
    
    assigned_away_rate = np.array([[assigned_away_hpi, assigned_away_hpi1, assigned_away_hpi2, assigned_away_hpi3]])
    assigned_away_population = np.round(assigned_away_rate * population_vec / 1000000,4)[0]
    assigned_away_rate = np.round(assigned_away_rate * 100)[0]


    ## Total vaccination
    total_rate_hpi = np.sum(np.multiply(F_DH, mat_y_hpi), axis = 1)
    CA_TRACT['Rate_HPI'] = total_rate_hpi
    CA_TRACT['Vaccinated_Population_HPI'] = CA_TRACT['Rate_HPI'] * CA_TRACT['Population']
    
    total_rate_hpi = sum(CA_TRACT['Vaccinated_Population_HPI'].values) / total_population
    total_rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccinated_Population_HPI'].values) / population1
    total_rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccinated_Population_HPI'].values) / population2
    total_rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccinated_Population_HPI'].values) / population3

    total_rate = np.array([[total_rate_hpi, total_rate_hpi1, total_rate_hpi2, total_rate_hpi3]])
    total_vaccination = np.round(total_rate * population_vec / 1000000,4)[0]
    total_proportion = np.round(total_rate * 100)[0]


    ## Vaccination within M
    rate_hpi = np.sum(np.multiply(F_DH, mat_y_hpi_closest), axis = 1)
    CA_TRACT['Rate_HPI'] = rate_hpi
    CA_TRACT['Vaccinated_Population_HPI'] = CA_TRACT['Rate_HPI'] * CA_TRACT['Population']
    
    rate_hpi = sum(CA_TRACT['Vaccinated_Population_HPI'].values) / total_population
    rate_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccinated_Population_HPI'].values) / population1
    rate_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccinated_Population_HPI'].values) / population2
    rate_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccinated_Population_HPI'].values) / population3
    
    rate = np.array([[rate_hpi, rate_hpi1, rate_hpi2, rate_hpi3]])
    vaccination = np.round(rate * population_vec / 1000000,4)[0]
    proportion = np.round(rate * 100)[0]
    

    ## Vaccination away M
    drop_hpi = np.sum(np.multiply(F_DH, mat_y_hpi_farthest), axis = 1)
    CA_TRACT['Drop_HPI'] = drop_hpi
    CA_TRACT['Drop_Vaccination_HPI'] = CA_TRACT['Drop_HPI'] * CA_TRACT['Population']
    
    drop_hpi = sum(CA_TRACT['Drop_Vaccination_HPI'].values) / total_population
    drop_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Drop_Vaccination_HPI'].values) / population1
    drop_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Drop_Vaccination_HPI'].values) / population2
    drop_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Drop_Vaccination_HPI'].values) / population3
    
    drop = np.array([[drop_hpi, drop_hpi1, drop_hpi2, drop_hpi3]])
    drop_vaccination = np.round(drop * population_vec / 1000000,4)[0]
    drop_proportion = np.round(drop * 100)[0]
    
    
    ## Assigned within Walkable
    assigned_walkable_hpi = np.sum(np.multiply(C_walkable, mat_y_hpi), axis=1) 
    CA_TRACT['Assigned_Walkable_HPI'] = assigned_walkable_hpi
    CA_TRACT['Assigned_Walkable_Population_HPI'] = CA_TRACT['Assigned_Walkable_HPI'] * CA_TRACT['Population']    

    assigned_walkable_hpi = sum(CA_TRACT['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Walkable_Population_HPI'].values)
    assigned_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Walkable_Population_HPI'].values)

    assigned_walkable_list = np.array([[assigned_walkable_hpi, assigned_walkable_hpi1, assigned_walkable_hpi2, assigned_walkable_hpi3]])
    assigned_walkable = np.round(assigned_walkable_list / 1000000, 4)[0]
    assigned_walkable_proportion = np.round(assigned_walkable_list[0] / population_vec * 100)


    ## Vaccination within Walkable
    rate_walkable_hpi = np.sum(np.multiply(C_walkable, np.multiply(F_DH, mat_y_hpi)), axis =1) 
    CA_TRACT['Vaccinate_Walkable_HPI'] = rate_walkable_hpi
    CA_TRACT['Vaccination_Walkable_HPI'] = CA_TRACT['Vaccinate_Walkable_HPI'] * CA_TRACT['Population']    

    rate_walkable_hpi = sum(CA_TRACT['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Vaccination_Walkable_HPI'].values)
    rate_walkable_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Vaccination_Walkable_HPI'].values)
    
    rate_walkable_list = np.array([[rate_walkable_hpi, rate_walkable_hpi1, rate_walkable_hpi2, rate_walkable_hpi3]])
    vaccination_walkable = np.round(rate_walkable_list / 1000000, 4)[0]
    rate_walkable = np.round(rate_walkable_list[0] / population_vec * 100)


    ## Distance among assigned
    assigned_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi), axis = 1) / np.sum(mat_y_hpi, axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Assigned'] = assigned_dist_hpi
    CA_TRACT['Assigned_Population_HPI'] = CA_TRACT['Population'] * np.sum(mat_y_hpi, axis = 1)
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Assigned'] * CA_TRACT['Assigned_Population_HPI']
    
    assigned_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Assigned_Population_HPI'].values)
    assigned_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Assigned_Population_HPI'])
    assigned_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Assigned_Population_HPI'])
    assigned_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Assigned_Population_HPI'])
    
    assigned_dist_actual = np.array([[assigned_dist_hpi, assigned_dist_hpi1, assigned_dist_hpi2, assigned_dist_hpi3]])
    assigned_dist_actual = np.round(assigned_dist_actual/1000, 4)[0]
    

    ## Distance among assigned within M
    closest_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi_closest), axis = 1) / np.sum(mat_y_hpi_closest, axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Closest'] = closest_dist_hpi
    CA_TRACT['Closest_Population_HPI'] = np.sum(mat_y_hpi_closest, axis = 1) * CA_TRACT['Population']
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Closest'] * CA_TRACT['Closest_Population_HPI']

    closest_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Closest_Population_HPI'].values)
    closest_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Closest_Population_HPI'])
    closest_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Closest_Population_HPI'])
    closest_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Closest_Population_HPI'])

    closest_dist_actual = np.array([[closest_dist_hpi, closest_dist_hpi1, closest_dist_hpi2, closest_dist_hpi3]])
    closest_dist_actual = np.round(closest_dist_actual/1000, 4)[0]
    

    ## Distance among assigned away M
    farthest_dist_hpi = np.nan_to_num(np.sum(np.multiply(C, mat_y_hpi_farthest), axis = 1) / np.sum(mat_y_hpi_farthest, axis = 1), posinf=0)
    CA_TRACT['Dist_HPI_Farthest'] = farthest_dist_hpi
    CA_TRACT['Farthest_Population_HPI'] = np.sum(mat_y_hpi_farthest, axis = 1) * CA_TRACT['Population']
    CA_TRACT['Dist_HPI_weighted'] = CA_TRACT['Dist_HPI_Farthest'] * CA_TRACT['Farthest_Population_HPI']

    farthest_dist_hpi = sum(CA_TRACT['Dist_HPI_weighted'].values) / sum(CA_TRACT['Farthest_Population_HPI'].values)
    farthest_dist_hpi1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Farthest_Population_HPI'])
    farthest_dist_hpi2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Farthest_Population_HPI'])
    farthest_dist_hpi3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Dist_HPI_weighted'].values) / sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Farthest_Population_HPI'])

    farthest_dist_actual = np.array([[farthest_dist_hpi, farthest_dist_hpi1, farthest_dist_hpi2, farthest_dist_hpi3]])
    farthest_dist_actual = np.round(farthest_dist_actual/1000, 4)[0]
    

    ###########################################################################

    chain_summary = {'Model': Model, 'Chain': Scenario,
                     'Opt Constr': opt_constr, 'Eval Constr': eval_constr,
                     'M': M, 'K': K,
                     'R': R,
                     'Vaccination': total_vaccination[0], 
                     'Vaccination HPI1': total_vaccination[1], 'Vaccination HPI2': total_vaccination[2], 
                     'Vaccination HPI3': total_vaccination[3],
                     'Rate': total_proportion[0],
                     'Rate HPI1': total_proportion[1], 'Rate HPI2': total_proportion[2], 
                     'Rate HPI3': total_proportion[3],
                     'Vaccination Within M': vaccination[0], 
                     'Vaccination Within M HPI1': vaccination[1], 'Vaccination Within M HPI2': vaccination[2], 
                     'Vaccination Within M HPI3': vaccination[3],
                     'Rate Within M': proportion[0],
                     'Rate Within M HPI1': proportion[1], 'Rate Within M HPI2': proportion[2], 
                     'Rate Within M HPI3': proportion[3],
                     'Assigned Within M': assigned_within_population[0], 
                     'Assigned Within M HPI1': assigned_within_population[1], 'Assigned Within M HPI2': assigned_within_population[2], 
                     'Assigned Within M HPI3': assigned_within_population[3],
                     'Assigned Rate Within M': assigned_within_rate[0], 
                     'Assigned Rate Within M HPI1': assigned_within_rate[1], 'Assigned Rate Within M HPI2': assigned_within_rate[2], 
                     'Assigned Rate Within M HPI3': assigned_within_rate[3],
                     'Assigned Walkable': assigned_walkable[0], 
                     'Assigned Walkable HPI1': assigned_walkable[1], 'Assigned Walkable HPI2': assigned_walkable[2], 
                     'Assigned Walkable HPI3': assigned_walkable[3],
                     'Assigned Walkable rate': assigned_walkable_proportion[0], 
                     'Assigned Walkable rate HPI1': assigned_walkable_proportion[1], 'Assigned Walkable rate HPI2': assigned_walkable_proportion[2],
                     'Assigned Walkable rate HPI3': assigned_walkable_proportion[3],
                     'Vaccination Walkable': vaccination_walkable[0], 
                     'Vaccination Walkable HPI1': vaccination_walkable[1], 'Vaccination Walkable HPI2': vaccination_walkable[2], 
                     'Vaccination Walkable HPI3': vaccination_walkable[3],
                     'Vaccination Walkable rate': rate_walkable[0], 
                     'Vaccination Walkable rate HPI1': rate_walkable[1], 'Vaccination Walkable rate HPI2': rate_walkable[2],
                     'Vaccination Walkable rate HPI3': rate_walkable[3],
                     'Average distance': assigned_dist_actual[0],
                     'Average distance HPI1': assigned_dist_actual[1], 'Average distance HPI2': assigned_dist_actual[2],
                     'Average distance HPI3': assigned_dist_actual[3],
                    }                   



    return chain_summary
   



def tract_summary(Model, Chain_type, M, K, Quartile, Population,\
mat_y_current_hpi, mat_y_current_hpi_closest, mat_y_current_hpi_farthest, result_current_hpi, F_DH_current, C_current, C_current_walkable,\
mat_y_total_hpi, mat_y_total_hpi_closest, mat_y_total_hpi_farthest, result_total_hpi, F_DH_total, C_total, C_total_walkable,\
expdirpath = "/export/storage_covidvaccine/Result"):


    datadir = "/export/storage_covidvaccine/Data"
    Tract_ID = np.genfromtxt(f'{datadir}/CA_tractID.csv', delimiter = ",", dtype = str)
    CA_TRACT = pd.DataFrame(data = {'GEOID': Tract_ID,'Population': Population, 'HPIQuartile': Quartile})

    total_population = sum(Population)
    population1 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 1]['Population'].values)
    population2 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 2]['Population'].values)
    population3 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 3]['Population'].values)
    population4 = sum(CA_TRACT[CA_TRACT['HPIQuartile'] == 4]['Population'].values)
    population_vec = [total_population, population1, population2, population3, population4]


 
    ## Current

    CA_TRACT['Assigned_Rate_Current'] = np.sum(mat_y_current_hpi, axis = 1)
    CA_TRACT['Assigned_Rate_WithinM_Current'] = np.round(np.sum(mat_y_current_hpi_closest, axis = 1),3)
    CA_TRACT['Assigned_Rate_AwayM_Current'] = np.round(np.sum(mat_y_current_hpi_farthest, axis = 1),3)

    CA_TRACT['Assigned_Population_Current'] = np.round(np.sum(mat_y_current_hpi, axis = 1) * CA_TRACT['Population'])
    CA_TRACT['Assigned_Population_WithinM_Current'] = np.round(np.round(np.sum(mat_y_current_hpi_closest, axis = 1),3) * CA_TRACT['Population'])
    CA_TRACT['Assigned_Population_AwayM_Current'] = np.round(np.round(np.sum(mat_y_current_hpi_farthest, axis = 1),3) * CA_TRACT['Population'])


    CA_TRACT['Rate_Current'] = np.round(np.sum(np.multiply(F_DH_current, mat_y_current_hpi), axis = 1),3)
    CA_TRACT['Vaccination_Current'] = np.round(CA_TRACT['Rate_Current'] * CA_TRACT['Population'])

    CA_TRACT['Rate_WithinM_Current'] = np.round(np.sum(np.multiply(F_DH_current, mat_y_current_hpi_closest), axis = 1),3)
    CA_TRACT['Vaccination_WithinM_Current'] = np.round(CA_TRACT['Rate_WithinM_Current'] * CA_TRACT['Population'])

    CA_TRACT['Rate_AwayM_Current'] = np.round(np.sum(np.multiply(F_DH_current, mat_y_current_hpi_farthest), axis = 1),3)
    CA_TRACT['Vaccination_AwayM_Current'] = np.round(CA_TRACT['Rate_AwayM_Current'] * CA_TRACT['Population'])
    
        
    CA_TRACT['Dist_Assigned_Current'] = np.round(np.nan_to_num(np.sum(np.multiply(C_current, mat_y_current_hpi), axis = 1) / np.sum(mat_y_current_hpi, axis = 1), posinf=0))
    CA_TRACT['Dist_Assigned_WithinM_Current'] = np.round(np.nan_to_num(np.sum(np.multiply(C_current, mat_y_current_hpi_closest), axis = 1) / np.sum(mat_y_current_hpi_closest, axis = 1), posinf=0))
    CA_TRACT['Dist_Assigned_AwayM_Current'] = np.round(np.nan_to_num(np.sum(np.multiply(C_current, mat_y_current_hpi_farthest), axis = 1) / np.sum(mat_y_current_hpi_farthest, axis = 1), posinf=0))

    CA_TRACT['Rate_Walkable_Current'] = np.round(np.sum(np.multiply(C_current_walkable, np.multiply(F_DH_current, mat_y_current_hpi)), axis =1),3)
    CA_TRACT['Vaccination_Walkable_Current'] = np.round(CA_TRACT['Rate_Walkable_Current'] * CA_TRACT['Population'])


    ## Total

    CA_TRACT['Assigned_Rate_Total'] = np.sum(mat_y_total_hpi, axis = 1)
    CA_TRACT['Assigned_Rate_WithinM_Total'] = np.round(np.sum(mat_y_total_hpi_closest, axis = 1),3)
    CA_TRACT['Assigned_Rate_AwayM_Total'] = np.round(np.sum(mat_y_total_hpi_farthest, axis = 1),3)

    CA_TRACT['Assigned_Population_Total'] = np.round(np.sum(mat_y_total_hpi, axis = 1) * CA_TRACT['Population'])
    CA_TRACT['Assigned_Population_WithinM_Total'] = np.round(np.round(np.sum(mat_y_total_hpi_closest, axis = 1),3) * CA_TRACT['Population'])
    CA_TRACT['Assigned_Population_AwayM_Total'] = np.round(np.round(np.sum(mat_y_total_hpi_farthest, axis = 1),3) * CA_TRACT['Population'])

    
    CA_TRACT['Rate_Total'] = np.round(np.sum(np.multiply(F_DH_total, mat_y_total_hpi), axis = 1),3)
    CA_TRACT['Vaccination_Total'] = np.round( CA_TRACT['Rate_Total'] * CA_TRACT['Population'])

    CA_TRACT['Rate_WithinM_Total'] = np.round(np.sum(np.multiply(F_DH_total, mat_y_total_hpi_closest), axis = 1),3)
    CA_TRACT['Vaccination_WithinM_Total'] = np.round(CA_TRACT['Rate_WithinM_Total'] * CA_TRACT['Population'])

    CA_TRACT['Rate_AwayM_Total'] = np.round(np.sum(np.multiply(F_DH_total, mat_y_total_hpi_farthest), axis = 1),3)
    CA_TRACT['Vaccination_AwayM_Total'] = np.round(CA_TRACT['Rate_AwayM_Total'] * CA_TRACT['Population'])


    CA_TRACT['Dist_Assigned_Total'] = np.round(np.nan_to_num(np.sum(np.multiply(C_total, mat_y_total_hpi), axis = 1) / np.sum(mat_y_total_hpi, axis = 1), posinf=0))
    CA_TRACT['Dist_Assigned_WithinM_Total'] = np.round(np.nan_to_num(np.sum(np.multiply(C_total, mat_y_total_hpi_closest), axis = 1) / np.sum(mat_y_total_hpi_closest, axis = 1), posinf=0))
    CA_TRACT['Dist_Assigned_AwayM_Total'] = np.round(np.nan_to_num(np.sum(np.multiply(C_total, mat_y_total_hpi_farthest), axis = 1) / np.sum(mat_y_total_hpi_farthest, axis = 1), posinf=0))


    CA_TRACT['Rate_Walkable_Total'] = np.round(np.sum(np.multiply(C_total_walkable, np.multiply(F_DH_total, mat_y_total_hpi)), axis =1),3)
    CA_TRACT['Vaccination_Walkable_Total'] = np.round(CA_TRACT['Rate_Walkable_Total'] * CA_TRACT['Population'])


    ## Stores
    num_tracts, num_current_stores = np.shape(C_current)
    num_tracts, num_total_stores = np.shape(C_total)
    CA_TRACT['Pharmacy_Used_Current'] = np.sum(np.ceil(mat_y_current_hpi[:, 0:num_current_stores]), axis = 1)
    CA_TRACT['Pharmacy_Used_Total'] = np.sum(np.ceil(mat_y_total_hpi[:, 0:num_current_stores]), axis = 1)
    CA_TRACT['ChainStore_Used_Total'] = np.sum(np.ceil(mat_y_total_hpi[:, num_current_stores:num_total_stores]), axis = 1)


    ###########################################################################

    ## Export
    CA_TRACT.to_csv(f'{expdirpath}/CA_TRACT_M' + str(M) + '_K' + str(K) + '_' + Model + '.csv', encoding='utf-8', index=False, header=True)

