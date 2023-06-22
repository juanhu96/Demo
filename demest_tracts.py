# Demand estimation with tract-based distances.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"


# config = [False, False, False]
# config = [True, False, False]
# config = [True, True, False]
# config = [True, True, True]


for config in [
    [False, False, False],
    [True, False, False],
    [True, True, False],
    [True, True, True]
    ]:

    include_hpiquartile, interact_disthpi, include_controls = config

    print(f"Running config: include_hpiquartile={include_hpiquartile}, interact_disthpi={interact_disthpi}, include_controls={include_controls}")
    
    df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
    df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables



    agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")

    list(df.columns)
    df.logdist.describe()
    agent_data.logdist.describe()


    if interact_disthpi:
        # agent_formulation = pyblp.Formulation('0 + logdistXhpi1 + logdistXhpi2 + logdistXhpi3 + logdistXhpi4')
        agent_formulation = pyblp.Formulation('0 + logdist:C(hpi_quartile)')
        pi_init = -0.1*np.ones((1,4))
    else:
        agent_formulation = pyblp.Formulation('0 + logdist')
        pi_init = -0.1*np.ones((1,1))


    print(agent_data.describe())

    if include_controls:
        controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
            'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
            'medianhomevalue', 'popdensity'] #, 'population', 'dshare']
        formula_str = "1 + prices +  " + " + ".join(controls)
    else:
        formula_str = "1 + prices"
    
    if include_hpiquartile:
        df = df.assign(
            hpi_quartile1 = (df['hpi_quartile'] == 1).astype(int),
            hpi_quartile2 = (df['hpi_quartile'] == 2).astype(int),
            hpi_quartile3 = (df['hpi_quartile'] == 3).astype(int)
        )
        formulation1 = pyblp.Formulation(formula_str + '+ hpi_quartile1 + hpi_quartile2 + hpi_quartile3')
    else:
        formulation1 = pyblp.Formulation(formula_str)
    
    formulation2 = pyblp.Formulation('1')

    if interact_disthpi:
        for qq in range(1,5):
            agent_data[f"logdistXhpi{qq}"] = agent_data['logdist'] * (agent_data['hpi_quartile'] == qq)
            demand_instruments_qq = pd.DataFrame({f'demand_instruments{qq-1}': agent_data.groupby('market_ids')[f'logdistXhpi{qq}'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
            df = df.merge(demand_instruments_qq, left_on='market_ids', right_index=True)
    else:
        df['demand_instruments0'] = agent_data.groupby('market_ids')['logdist'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights'])).reset_index(drop=True)

    problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                            product_data=df, 
                            agent_formulation=agent_formulation, 
                            agent_data=agent_data)
    print(problem)


    iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12})
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10, 'verbose':1})
    # iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-11}) #TODO: remove this line
    # optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-8, 'verbose':1}) #TODO: remove this line

    with pyblp.parallel(32):
        results = problem.solve(pi=pi_init,
                                iteration = iteration_config,
                                optimization = optimization_config,
                                sigma = 0
                                )

    print(results)
    results.to_pickle(f"{datadir}/Analysis/tracts_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")
