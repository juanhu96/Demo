# Demand estimation with tract-based distances.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"

# config = [False, False, False]
# config = [True, False, False]
config = [True, True, False]
# config = [True, True, True]


for config in [
    [False, False, False],
    [True, False, False],
    [True, True, False],
    [True, True, True]
    ]:

    include_hpiquartile, interact_disthpi, include_controls = config

    print(f"Running config: include_hpiquartile={include_hpiquartile}, interact_disthpi={interact_disthpi}, include_controls={include_controls}")
    
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables

    agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")

    agent_data.describe()
    
    if interact_disthpi:
        agent_data = agent_data.assign(
            logdistXhpi_quartile1 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 1),
            logdistXhpi_quartile2 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 2),
            logdistXhpi_quartile3 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 3),
            logdistXhpi_quartile4 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 4))
        agent_formulation = pyblp.Formulation('0 + logdistXhpi_quartile1 + logdistXhpi_quartile2 + logdistXhpi_quartile3 + logdist')
        # agent_formulation = pyblp.Formulation('0 + logdistXhpi_quartile1 + logdistXhpi_quartile2 + logdistXhpi_quartile3 + logdistXhpi_quartile4')
        pi_init = -0.001*np.ones((1,4))
    else:
        agent_formulation = pyblp.Formulation('0 + logdist')
        pi_init = -0.001*np.ones((1,1))

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
            hpi_quartile3 = (df['hpi_quartile'] == 3).astype(int))
        formulation1 = pyblp.Formulation(formula_str + '+ hpi_quartile1 + hpi_quartile2 + hpi_quartile3')
    else:
        formulation1 = pyblp.Formulation(formula_str)
    
    formulation2 = pyblp.Formulation('1')

    # Instruments
    if interact_disthpi:
        # (a) distance * hpi_quartile IVs
        for qq in range(1,5):
            agent_data[f"logdistXhpi{qq}"] = agent_data['logdist'] * (agent_data['hpi_quartile'] == qq)
            demand_instruments_qq = pd.DataFrame({f'demand_instruments{qq-1}': agent_data.groupby('market_ids')[f'logdistXhpi{qq}'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
            df = df.merge(demand_instruments_qq, left_on='market_ids', right_index=True)
    else:
        df['demand_instruments0'] = agent_data.groupby('market_ids')['logdist'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights'])).reset_index(drop=True)

    problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)

    iteration_config = pyblp.Iteration(method='lm')
    # iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-13})
    tighter_tols = True #TODO: switch
    if tighter_tols:
        optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-12})
    else:
        optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-8})


    with pyblp.parallel(32):
        results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)


    results.to_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")


# export results to csv

include_hpiquartile, interact_disthpi, include_controls = [False, False, False]
results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")
m1coefs = np.array([results.beta[0][0], results.pi[0][0]])


include_hpiquartile, interact_disthpi, include_controls = [True, True, False]
results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")


m2coefs = np.array([results.beta[0][0], results.pi[0][3], results.beta[2][0], results.beta[3][0], results.beta[4][0], results.pi[0][0], results.pi[0][1], results.pi[0][2]])

useold = False #TODO:
if useold:
    np.save(f'{datadir}/Analysis/m1coefs_0622.npy', m1coefs)
    np.save(f'{datadir}/Analysis/m2coefs_0622.npy', m2coefs)
else:
    np.save(f'{datadir}/Analysis/m1coefs.npy', m1coefs)
    np.save(f'{datadir}/Analysis/m2coefs.npy', m2coefs)

m1coefs = np.load(f'{datadir}/Analysis/m1coefs.npy')
m2coefs = np.load(f'{datadir}/Analysis/m2coefs.npy')
print("Coefficients exported:")
print("[constant, log(dist)]  from column 1:") 
print(m1coefs)
print("[constant, log(dist), HPI quartile 1, HPI quartile 2, HPI quartile 3, HPI quartile 1 * log(dist), HPI quartile 2 * log(dist), HPI quartile 3 * log(dist)]\nfrom column 3:")
print(m2coefs)

# print("Old coefficients:")
# m1coefs = np.load(f'{datadir}/Analysis/m1coefs_0622.npy')
# m2coefs = np.load(f'{datadir}/Analysis/m2coefs_0622.npy')
# print(m1coefs)
# print(m2coefs)