# Demand estimation with tract-based distances
# Superceded by demest_tracts_rc.py
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

df = pd.read_csv(f"{datadir}/Analysis/product_data_tracts.csv")
agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")

list(agent_data.columns)
agent_data['logdist'] = np.log(agent_data['dist'])

# MAKE DISTANCE BY HPIQUARTILE
for qq in range(1,5):
    agent_data[f"logdistXhpi{qq}"] = agent_data['logdist'] * (agent_data['hpiquartile'] == qq)
    agent_data[f"logdistXhpi{qq}"].describe()

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
            'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
            'medianhomevalue', 'popdensity', 'population']
formula_str = "1 + prices +  " + " + ".join(controls)
formulation1 = pyblp.Formulation(formula_str + '+ C(hpiquartile)')
formulation2 = pyblp.Formulation('1')
agent_formulation = pyblp.Formulation('0 + logdistXhpi1 + logdistXhpi2 + logdistXhpi3 + logdistXhpi4')

# ADD INSTRUMENT
# population weighted average log distance
logdist_popw = agent_data.groupby('market_ids')['dist'].apply(lambda x: np.average(np.log(x), weights=agent_data.loc[x.index, 'weights']))
for qq in range(1,5):
    df[f'demand_instruments{qq-1}'] = logdist_popw[df['market_ids']].values * (df['hpiquartile'] == qq)

ivcols = [cc for cc in df.columns if 'demand_instruments' in cc]
df[ivcols].describe()

# TODO: could also try the log of the average distance, just add

problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                        product_data=df, 
                        agent_formulation=agent_formulation, 
                        agent_data=agent_data)
print(problem)

iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12, 'max_evaluations':10000})
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-9, 'verbose':1, 'maxiter':600})

with pyblp.parallel(32):
    results = problem.solve(pi=-0.1*np.ones((1,4)),
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )

print(results)

###################
# Margins plots
###################

df['xi_fe'] = results.xi_fe #all zeros
df['xi'] = results.xi

# add agent component of utility
agent_mktids = results.problem.agents.market_ids.flatten()

# idf = pd.DataFrame({'u_i':u_i.flatten(), 'market_ids':agent_mktids})
idf = agent_data[['market_ids', 'weights']]
idf
# add the hpiquartile-specific distance coefficients
dist_coefs = results.pi.flatten()
idf['distbeta'] = 0
for qq in range(1,5):
    agent_data[f"is_hpi{qq}"] = agent_data['hpiquartile'] == qq
    idf['distbeta'] += agent_data[f"is_hpi{qq}"] * dist_coefs[qq-1] 


# mean utility (zip-level)
betas = results.beta.flatten()
results.beta_labels
betas.shape
whichbetasincontrol = np.where([cc in controls for cc in results.beta_labels])[0]
df['meanutil'] = np.dot(df[controls], betas[whichbetasincontrol])
# add constant
df['meanutil'] = df['meanutil'] + betas[0]
# add the hpiquartile fixed effects
results.beta_labels[-3:]
betas_fe = betas[-3:]
for qq in range(2, 5):
    df[f'hpiquartile{qq}'] = (df['hpiquartile'] == qq)
    df['meanutil'] = df['meanutil'] + df[f'hpiquartile{qq}'] * betas_fe[qq-2]

df['meanutil'] += df['xi_fe'] + df['xi']

df_marg = df.merge(idf, on='market_ids', how='right')
df_marg['weights'] = df_marg['population'] / df_marg['weights']

df_marg.drop(columns=['market_ids'], inplace=True)
df_marg.to_stata(f"{datadir}/Analysis/tracts_marg.dta")