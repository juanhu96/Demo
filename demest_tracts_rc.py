# run after prep_tracts.py. with RCs on distance
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3
np.random.seed(123)

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

df = pd.read_csv(f"{datadir}/Analysis/product_data_tracts.csv")
agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")

agent_data['logdist'] = np.log(agent_data['dist'])
# MAKE DISTANCE BY HPIQUARTILE
for qq in range(1,5):
    agent_data[f"logdistXhpi{qq}"] = agent_data['logdist'] * (agent_data['hpiquartile'] == qq)
    agent_data[f"logdistXhpi{qq}"].describe()

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other', 'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 'medianhomevalue', 'popdensity', 'population']

# draw nI from standard normal
nI = 50
nu_dist_t = np.random.normal(size=(nI))
nu_dist = np.tile(nu_dist_t, agent_data.shape[0])

# draw RC for the constant
nu_const_t = np.random.normal(size=(nI))
nu_const = np.tile(nu_const_t, agent_data.shape[0])


# repeat the agent data nI times
agent_data = agent_data.loc[np.repeat(agent_data.index.values, nI)].reset_index(drop=True)
agent_data['weights'] = agent_data['weights'] / nI
agent_data['nu_dist'] = nu_dist
# pre-multiply logdist by nu, so the estimated pi will be the variance
agent_data['nuXlogdist'] = agent_data['nu_dist'] * agent_data['logdist']
agent_data['nu_const'] = nu_const

df.to_csv(f"{datadir}/Analysis/product_data_tracts_rc.csv", index=False)
agent_data.to_csv(f"{datadir}/Analysis/agent_data_tracts_rc.csv", index=False)

formula_str = "1 + prices +  " + " + ".join(controls)
formulation1 = pyblp.Formulation(formula_str + '+ C(hpiquartile)')
formulation2 = pyblp.Formulation('1')
agent_formulation = pyblp.Formulation('0 + logdistXhpi1 + logdistXhpi2 + logdistXhpi3 + logdistXhpi4 + nuXlogdist + nu_const')

# ADD INSTRUMENT
# population-weighted quadratic distance
dist_popw = agent_data.groupby('market_ids')['dist'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))
dist2_popw = agent_data.groupby('market_ids')['dist'].apply(lambda x: np.average(x**2, weights=agent_data.loc[x.index, 'weights']))

for qq in range(1,5):
    df[f'demand_instruments{qq-1}'] = dist_popw[df['market_ids']].values * (df['hpiquartile'] == qq)
    df[f'demand_instruments{qq+3}'] = dist2_popw[df['market_ids']].values * (df['hpiquartile'] == qq)

ivcols = [cc for cc in df.columns if 'demand_instruments' in cc]
df[ivcols].describe()


problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                        product_data=df, 
                        agent_formulation=agent_formulation, 
                        agent_data=agent_data)
print(problem)

iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12, 'max_evaluations':10000})
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-9, 'verbose':1, 'maxiter':600})


with pyblp.parallel(32):
    results = problem.solve(pi=-0.1*np.ones((1,6)),
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )
results.to_pickle(f"{datadir}/Analysis/results_tracts_rc.pkl")

print(results)

###################
# Margins plots
###################

results = pyblp.read_pickle(f"{datadir}/Analysis/results_tracts_rc.pkl")

##
df = pd.read_csv(f"{datadir}/Analysis/product_data_tracts_rc.csv")
agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data_tracts_rc.csv")

df['xi'] = results.xi

# add agent component of utility
agent_mktids = results.problem.agents.market_ids.flatten()

idf = agent_data[['market_ids', 'hpiquartile', 'nu_dist', 'nu_const', 'weights']].copy()
# add the hpiquartile-specific distance coefficients
dist_coefs = results.pi.flatten()[0:4]
nuXlogdist_coef = results.pi.flatten()[4]
nu_const_coef = results.pi.flatten()[5]
print(results.pi_labels)
idf['distbeta'] = 0
for qq in range(1,5):
    agent_data[f"is_hpi{qq}"] = agent_data['hpiquartile'] == qq
    idf['distbeta'] += agent_data[f"is_hpi{qq}"] * dist_coefs[qq-1]
idf.distbeta.value_counts()

idf['u_i_const'] = nu_const_coef * idf['nu_const']


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

# add the unobserved quality
df['meanutil'] = df['meanutil'] + df['xi']

# merge product and agent data
idf = idf.rename(columns={'hpiquartile': 'hpiquartile_tract'})
df = df.rename(columns={'hpiquartile': 'hpiquartile_zip'})
list(df.columns)
df_marg = df[['zip', 'vaxfull', 'shares', 'market_ids', 'meanutil', 'population']].merge(idf, on='market_ids', how='right')
df_marg['meanutil'] += df_marg['u_i_const']
df_marg = df_marg.drop(columns=['u_i_const', 'nu_const'])
df_marg['coef_nuXlogdist'] = nuXlogdist_coef
df_marg['weights'] = df_marg['population'] / df_marg['weights']
list(df_marg.columns)
df_marg.describe()
df_marg.to_stata(f"{datadir}/Analysis/tracts_marg_rc.dta")

