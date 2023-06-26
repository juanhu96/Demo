# Demand estimation with tract-based distances. By-quartile specifications.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np
import sys

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"

for spec in ['hpi', 'dshare', 'race', 'income']:
    df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables

    agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")

    print(f"Running {spec} specification")

    if spec == 'distbin':
        dist_cuts = [0,1,3,5,7,10,15,20,max(agent_data.dist)]
        agent_data['distbin'] = pd.cut(agent_data.dist, bins=dist_cuts, labels=False) + 1
        agent_data = pd.concat([agent_data, pd.get_dummies(agent_data['distbin'], prefix='distbin')], axis=1)
        agent_formulation = pyblp.Formulation('0 + distbin_1 + distbin_2 + distbin_3 + distbin_4 + distbin_5 + distbin_6 + distbin_7')
    elif spec == 'pooled':
        agent_formulation = pyblp.Formulation('0 + logdist')
        agent_data = agent_data.assign(pooled = 1)

    if spec in ['hpi', 'dshare', 'race', 'income']:
        byvar = f"{spec}_quartile"
        for qq in range(1,5):
            agent_data[byvar] = pd.qcut(agent_data[spec], 4, labels=False) + 1
            agent_data[f"logdistX{spec}{qq}"] = agent_data['logdist'] * (agent_data[byvar] == qq)
            df[f'demand_instruments{qq-1}'] = agent_data.groupby('market_ids')[f'logdistX{spec}{qq}'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights'])).reset_index(drop=True)
        pi_init = -0.1*np.ones((1,4))
        agent_formulation = pyblp.Formulation(f'0 + logdistX{spec}1 + logdistX{spec}2 + logdistX{spec}3 + logdistX{spec}4')
    elif spec == 'distbin':
        for qq in range(1,8):
            df[f'demand_instruments{qq-1}'] = agent_data.groupby('market_ids')[f'distbin_{qq}'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights'])).reset_index(drop=True)
        pi_init = -0.1*np.ones((1,len(dist_cuts)-2))
        byvar = 'distbin'
    elif spec == 'pooled':
        df['demand_instruments0'] = agent_data.groupby('market_ids')['logdist'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights'])).reset_index(drop=True)
        pi_init = -0.1*np.ones((1,1))
        byvar = 'pooled'


    print(agent_data.describe())

    controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
                'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
                'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
                'medianhomevalue', 'popdensity', 'population', 'dshare']

    formula_str = "1 + prices +  " + " + ".join(controls)
    formulation1 = pyblp.Formulation(formula_str + '+ C(hpi_quartile)')
    formulation2 = pyblp.Formulation('1')

    ivcols = [cc for cc in df.columns if 'demand_instruments' in cc]
    print(df[ivcols].describe())

    problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                            product_data=df, 
                            agent_formulation=agent_formulation, 
                            agent_data=agent_data)
    print(problem)

    iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12})
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10, 'verbose':1})

    with pyblp.parallel(32):
        results = problem.solve(pi=pi_init,
                                iteration = iteration_config,
                                optimization = optimization_config,
                                sigma = 0
                                )

    print(results)
    results.to_pickle(f"{datadir}/Analysis/Demand/tracts_results_by{spec}.pkl")

    # output pi to csv
    if spec == 'hpi':
        pi = pd.DataFrame({'hpi_quartile':list(range(1,5)), 'pi':results.pi.flatten()})
        pi.to_csv(f"{datadir}/Analysis/Demand/tracts_pi_byhpi.csv", index=False)

    ###################
    # Margins plots
    ###################

    # load results
    results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/tracts_results_by{spec}.pkl")
    idf = agent_data[['market_ids', 'weights', byvar]]

    # add the quartile-specific distance coefficients
    dist_coefs = results.pi.flatten()
    dist_coefs_se = results.pi_se.flatten()

    idf = idf.assign(distbeta = 0, distbeta_se = 0)
    for qq in set(idf[byvar]):
        qq = int(qq)
        is_q = idf[byvar] == qq
        idf['distbeta'] += is_q * dist_coefs[qq-1] 
        idf['distbeta_se'] += is_q * dist_coefs_se[qq-1]

    # mean utility (zip-level)
    df['meanutil'] = results.compute_delta(market_id = df['market_ids'])

    df_marg = df[['market_ids', 'meanutil', 'population']].merge(idf, on='market_ids', how='right')
    df_marg['weights'] = df_marg['population'] * df_marg['weights']

    df_marg.sort_values(by=['market_ids'], inplace=True)
    df_marg.to_stata(f"{datadir}/Analysis/Demand/tracts_marg_by{spec}.dta")


