# Demand estimation with tract-based distances. Distance in bins. 
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables

agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")

print(f"Running distbin specification")


dist_cuts = [0,1,3,5,7,10,15,20,max(agent_data.dist)]
agent_data['distbin'] = pd.cut(agent_data.dist, bins=dist_cuts, labels=False) + 1
agent_data = pd.concat([agent_data, pd.get_dummies(agent_data['distbin'], prefix='distbin')], axis=1)
agent_formulation = pyblp.Formulation('0 + distbin_1 + distbin_2 + distbin_3 + distbin_4 + distbin_5 + distbin_6 + distbin_7')

for qq in range(1,8):
    df[f'demand_instruments{qq-1}'] = agent_data.groupby('market_ids')[f'distbin_{qq}'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights'])).reset_index(drop=True)
pi_init = -0.1*np.ones((1,len(dist_cuts)-2))


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
results.to_pickle(f"{datadir}/Analysis/tracts_results_distbin.pkl")

###################
# Margins plots #TODO:
###################
