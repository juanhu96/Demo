# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12, 'max_evaluations':10000})
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-9, 'verbose':1, 'maxiter':600})

agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")
df = pd.read_csv(f"{datadir}/Analysis/product_data_tracts.csv")

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
            'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
            'medianhomevalue', 'popdensity', 'population']

formula_str = "1 + prices +  " + " + ".join(controls)
formulation1 = pyblp.Formulation(formula_str + '+ C(hpiquartile)')
formulation2 = pyblp.Formulation('1')
agent_formulation = pyblp.Formulation('0+log(dist)')

# ADD INSTRUMENT
# population weighted average distance
dist_popw = agent_data.groupby('market_ids')['dist'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))
dist2_popw = dist_popw**2

# population weighted average log distance
logdist_popw = agent_data.groupby('market_ids')['dist'].apply(lambda x: np.average(np.log(x), weights=agent_data.loc[x.index, 'weights']))


iv_config = 'quad'
if iv_config == 'quad':
    df['demand_instruments0'] = dist_popw[df['market_ids']].values
    df['demand_instruments1'] = dist2_popw[df['market_ids']].values
elif iv_config == 'log':
    df['demand_instruments0'] = logdist_popw[df['market_ids']].values
    if 'demand_instruments1' in df.columns:
        df = df.drop(columns=['demand_instruments1'])

problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                        product_data=df, 
                        agent_formulation=agent_formulation, 
                        agent_data=agent_data)
print(problem)

with pyblp.parallel(32):
    results = problem.solve(pi=-0.1,
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )
    
agent_data['dist'].describe()


pd.Series(np.log(df['dist'])).describe()
pd.Series(np.log(agent_data['dist'])).describe()

# compute average tract-pharmacy distance by ZIP
agent_data = agent_data.assign(logdist = np.log(agent_data['dist']))
agent_data.groupby('market_ids')['dist'].mean().describe()

# mean of log distance
agent_data.groupby('market_ids')['logdist'].mean().describe()

# log of mean distance
np.log(agent_data.groupby('market_ids')['dist'].mean()).describe()

agent_data.groupby('market_ids')['dist'].count().describe()

####################
#### DISTANCE BY HPIQUARTILE ####
####################

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

# TODO: could also try the log of the average distance, just add

# other possible RCs: the intercept, and the log(dist) in general
# to have an RC on log(dist), we have to draw 

ivcols = [cc for cc in df.columns if 'demand_instruments' in cc]
df[ivcols].describe()


problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                        product_data=df, 
                        agent_formulation=agent_formulation, 
                        agent_data=agent_data)
print(problem)

with pyblp.parallel(32):
    results = problem.solve(pi=0.1*np.ones((1,4)),
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )

pyblp.options.digits = 3
print(results)


# Margins plots
deltas = results.compute_delta().flatten()
df['xi_fe'] = results.xi_fe
df['xi'] = results.xi

# add agent component of utility
u_i = results.problem.agents.demographics @ results.pi.transpose()
agent_mktids = results.problem.agents.market_ids.flatten()

idf = pd.DataFrame({'u_i':u_i.flatten(), 'market_ids':agent_mktids})
idf

dist_coefs = results.pi.flatten()
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

df['meanutil']