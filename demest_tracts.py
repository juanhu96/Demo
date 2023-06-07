import pandas as pd

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data

# Original ZIP-level data with vaccination rates (product_data)
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv", dtype={'zip': str})
df = df.assign(market_ids = df['zip'], prices = 0)

# Tract (geoid) level data with HPI and HPIQuartile
tract_hpi_df = pd.read_csv(f"{datadir}/tracts/HPItract.csv", dtype={'geoid': str})
tract_hpi_df
tract_hpi_df.rename(columns={'geoid': 'tract'}, inplace=True)
tract_hpi_df['tract'].apply(len).value_counts() # all 10-digits that start with 6

# Tract-ZIP crosswalk
tractzip_cw = pd.read_csv(f"{datadir}/tracts/tract_zip_032022.csv", usecols=['TRACT', 'ZIP'], dtype={'TRACT': str, 'ZIP': str})
# verify that TRACT is 11 digits
tractzip_cw['TRACT'].apply(len).value_counts() # all 11
tractzip_cw = tractzip_cw.assign(statefips = tractzip_cw['TRACT'].str[:3])
# keep CA only
tractzip_cw = tractzip_cw[tractzip_cw['statefips'] == '060']
tractzip_cw = tractzip_cw.assign(countyfips = tractzip_cw['TRACT'].str[3:6])
tractzip_cw['countyfips'] = tractzip_cw['countyfips'].apply(lambda x: x.zfill(4))
tractzip_cw = tractzip_cw.assign(tractid = tractzip_cw['TRACT'].str[6:])
# Make the Tract column the same as the one in tract_nearest_df (10 digits, start with 6)
tractzip_cw = tractzip_cw.assign(tract = '6' + tractzip_cw['countyfips'] + tractzip_cw['tractid'])
tractzip_cw = tractzip_cw[['tract', 'ZIP']]

# 2019 ACS tract demographics (has a ton of variables)
acs_df = pd.read_csv(f"{datadir}/tracts/CA_TRACT_demographics.csv", low_memory=False)
acs_df.rename(columns={'GIDTR': 'tract'}, inplace=True)
demog_cols = [cc for cc in acs_df.columns if 'pct' not in cc and 'avg' not in cc]
# demog_cols 
pop_cols = [cc for cc in demog_cols if 'pop' in cc or 'Pop' in cc]
# pop_cols
acs_df['tract'] = acs_df['tract'].astype(str)
acs_df['tract'].apply(len).value_counts() # all 10-digits that start with 6
tract_demog = acs_df[['tract']]
tract_demog = tract_demog.assign(trpop = acs_df['Tot_Population_ACS_14_18'])

# read tract-pharmacy distance pairs
# pre-processed to the 10 nearest pharmacies for each tract.
pairs_df = pd.read_csv(f"{datadir}/tracts/pairs_filtered.csv", usecols=['Tract', 'Distance'],dtype={'Tract': str, 'Distance': float})
pairs_df
pairs_df.rename(columns={'Tract': 'tract', 'Distance': 'dist'}, inplace=True)
# just the nearest pharmacy for each tract
tract_nearest_df = pairs_df.groupby('tract').head(1)

tract_nearest_df['tract'].apply(len).value_counts() # between 8 and 11
# The tract column is messed up. I think there should be FIPS as the first 5, with only the first digit being the state (6XXXX). Followed by a 5 digit tract ID. 
# TODO: verify if my fix is correct
# look at some examples 
tract_nearest_df.groupby(tract_nearest_df['tract'].apply(len)).apply(lambda x: x.sample(10))
# check that the first digit is always 0
tract_nearest_df['tract'].apply(lambda x: x[0]).value_counts()
tract_nearest_df = tract_nearest_df.assign(countyfips = tract_nearest_df['tract'].str[1:6])
tract_nearest_df['countyfips']
tract_nearest_df['tractid'] = tract_nearest_df['tract'].str[6:]
tract_nearest_df['tractid']
# pad the tractid with 0s
tract_nearest_df['tractid'] = tract_nearest_df['tractid'].apply(lambda x: x.zfill(5))
# combine the countyfips and tractid
tract_nearest_df['tract'] = tract_nearest_df['countyfips'] + tract_nearest_df['tractid']

# merge tract level data
tract_df = tract_nearest_df.merge(tract_hpi_df, on='tract', how='outer', indicator=True)
tract_df._merge.value_counts() #6k left, 5k right, 3k both
tract_df = tract_df.loc[tract_df._merge == 'both', :]
tract_df.drop(columns=['_merge'], inplace=True)

# merge with tract demographics (just population for now)
tract_df = tract_df.merge(tract_demog, on='tract', how='outer', indicator=True)
tract_df._merge.value_counts() # 5k right, 3k both
tract_df = tract_df.loc[tract_df._merge == 'both', :]
tract_df.drop(columns=['_merge'], inplace=True)

# merge tract_df with tractzip_cw
agent_data = tractzip_cw.merge(tract_df, on='tract', how='outer', indicator=True)
agent_data._merge.value_counts() # 300 left, 6k right, 14k both
agent_data = agent_data.loc[agent_data._merge == 'both', :]
agent_data.drop(columns=['_merge'], inplace=True)
agent_data = agent_data.rename(columns={'ZIP': 'zip', 'HPI': 'hpi', 'HPIQuartile': 'hpiquartile'})
# 592 zips in df aren't in agent_data
df['zip'].isin(agent_data['zip']).value_counts()

###### 
# get the agent_data into pyblp format

list(df.columns)
list(agent_data.columns)

# If a ZIP has no tracts, create a fake tract that's just the ZIP's HPI and population
aux_tracts = df[['hpi', 'hpiquartile', 'dist', 'market_ids']][~df['zip'].isin(agent_data['zip'])]
aux_tracts = aux_tracts.assign(weights = 1)
aux_tracts

# weights to be the population of the tract over the sum of the population of all tracts in the ZIP
zip_pop = agent_data.groupby('zip')['trpop'].transform('sum')

agent_data = agent_data.assign(market_ids = agent_data['zip'],
                               weights = agent_data['trpop']/zip_pop)

agent_data = agent_data[['market_ids', 'weights', 'hpi', 'hpiquartile', 'dist']]
agent_data = pd.concat([agent_data, aux_tracts], ignore_index=True)

# keep ZIPs that are in df
agent_data = agent_data[agent_data['market_ids'].isin(df['market_ids'])]
agent_data['nodes'] = 0 

# agent_data = agent_data.rename(columns={'dist': 'dist0'})
# save to csv
agent_data.to_csv(f"{datadir}/Analysis/agent_data.csv", index=False)
df.to_csv(f"{datadir}/Analysis/product_data_tracts.csv", index=False)


######## PYBLP ########
import pyblp
import pandas as pd
import numpy as np


iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12, 'max_evaluations':10000})
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10, 'verbose':1, 'maxiter':600})


datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")
df = pd.read_csv(f"{datadir}/Analysis/product_data_tracts.csv")

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
            'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
            'medianhomevalue', 'popdensity', 'population']
formula_str = "1 + prices +  " + " + ".join(controls)
formulation1 = pyblp.Formulation(formula_str + '+ C(hpiquartile)')
formulation2 = pyblp.Formulation('0+log(dist)')
agent_formulation = pyblp.Formulation('0+log(dist)')
pi_init = -0.1

problem = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                        product_data=df, 
                        agent_formulation=agent_formulation, 
                        agent_data=agent_data)
print(problem)

####
# agent_data2[agent_data.market_ids == 95129]


####

with pyblp.parallel(32):
    results = problem.solve(pi=pi_init,
                            error_punishment=3,
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )

dist_elast = results.compute_elasticities('dist')
pd.Series(dist_elast.flatten()).describe()

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
#### FIX ALL DISTANCES TO ZIP LEVEL DISTANCES ####
####################
# replace all tract-pharmacy distances with the distance for the ZIP
agent_data2 = agent_data.copy()
agent_data2 = agent_data2.drop(columns=['dist'])
agent_data2 = agent_data2.merge(df[['market_ids', 'dist']], on='market_ids', how='left', indicator=True)
agent_data2._merge.value_counts() #all merged
agent_data2.drop(columns=['_merge'], inplace=True)
agent_data2 = agent_data2.rename(columns={'dist': 'dist0'})

problem2 = pyblp.Problem(product_formulations=(formulation1, formulation2), 
                        product_data=df, 
                        agent_formulation=agent_formulation, 
                        agent_data=agent_data2)
print(problem2)


with pyblp.parallel(32):
    results2 = problem2.solve(pi=pi_init,
                            error_punishment=3,
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )


####################
# replace all tract data with ZIP data
####################

agent_data3 = df[['market_ids', 'hpi', 'hpiquartile', 'dist']]
agent_data3 = agent_data3.assign(weights = 1)
agent_data3 = agent_data3.rename(columns={'dist': 'dist0'})
agent_data3['nodes'] = 0

problem3 = pyblp.Problem(product_formulations=(formulation1, formulation2),
                        product_data=df,
                        agent_formulation=agent_formulation,
                        agent_data=agent_data3)
print(problem3)

with pyblp.parallel(32):
    results3 = problem3.solve(pi=pi_init,
                            error_punishment=3,
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )
    
####################TODO:
# TRY LOGIT
# mwe - one control, log dist, match with logit 
# simple data - minimal code


logit_formulation = pyblp.Formulation(formula_str + '+ log(dist)*C(hpiquartile)')
logit_formulation
logit_problem = pyblp.Problem(logit_formulation, df)
logit_results = logit_problem.solve()
logit_results
# verified that logit matched with stata

# simple logit
logit_formulation2 = pyblp.Formulation('1 + prices + collegegrad + log(dist)')
logit_formulation2
logit_problem2 = pyblp.Problem(logit_formulation2, df)
logit_results2 = logit_problem2.solve()
logit_results2

# logit with distance in the agent data
logit_formulation3 = (pyblp.Formulation('1 + prices + collegegrad'), pyblp.Formulation('0 + log(dist)'))
logit_agent_formulation = pyblp.Formulation('0 + log(dist)')
logit_agent_data = df[['market_ids', 'dist']]
logit_agent_data = logit_agent_data.assign(nodes = 0, weights = 1)
logit_problem3 = pyblp.Problem(logit_formulation3, df, logit_agent_formulation, logit_agent_data)
logit_results3 = logit_problem3.solve(pi=-0.1, error_punishment=3, iteration=iteration_config, optimization=optimization_config, sigma=0)
logit_results3


# # try with auxiliary ones column
# df['aux1'] = 1
# logit_formulation3 = (pyblp.Formulation('1 + prices + collegegrad'), pyblp.Formulation('0 + aux1'))
# logit_agent_formulation = pyblp.Formulation('0 + log(dist)')
# logit_agent_data = df[['market_ids', 'dist']]
# logit_agent_data = logit_agent_data.assign(nodes = 0, weights = 1)
# logit_problem3 = pyblp.Problem(logit_formulation3, df, logit_agent_formulation, logit_agent_data)
# logit_results3 = logit_problem3.solve(pi=-0.1, error_punishment=3, iteration=iteration_config, optimization=optimization_config, sigma=0)
# logit_results3






####################
####################
####################
#######  LOG OBJECTIVES MESH
####################

import matplotlib.pyplot as plt

pyblp.options.verbose = False
pyblp.options.verbose_tracebacks = False

lb = -2
ub = 0
mesh = np.linspace(lb, ub, num=int(round(2*(ub-lb)+1)))

obj_log = []
pp_log = []
for pp in mesh:
    try:
        with pyblp.parallel(32):
            results3 = problem3.solve(pi=pi_init, #use the ZIP-level data
                            error_punishment=3,
                            iteration = iteration_config,
                            optimization = optimization_config,
                            sigma = 0
                            )
            obj = results3.objective
            objval = obj[0][0]
            print("NL coefficient:", pp, "  Objective:", objval)
            obj_log.append(objval)
            pp_log.append(pp)
    except:
        continue


fig = plt.figure()
ax = plt.subplot(111)

ax.scatter(pp_log, obj_log)

fig.savefig("/mnt/staff/zhli/vaxobj.pdf", dpi=150)
