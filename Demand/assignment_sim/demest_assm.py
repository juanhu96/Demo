# demand estimation with capacity constraints

# 0. Draw a random order for demanders once.
# 1. Estimate the demand model under a given matching function (start with closest facility and no capacity constraints)
# 2. Apply estimated demand to the random order of demanders, except the distance for each demander is based on the closest location with available capacity by the time they get their choice.
# 3. Repeat step 1 assuming the location matching from step 2. Repeat iteratively until you reach a fixed point


# Notation: I use "geog" to denote a block/tract. We're using blocks now.

import pandas as pd
import numpy as np
import time
import importlib

import sys
sys.path.append("/mnt/staff/zhli/VaxDemandDistance/Demand/assignment_sim/") #TODO: only need this when running in terminal

from vax_entities import Individual, Geog
from assignment_funcs import initialize_geogs, compute_ranking, random_fcfs, reset_assignments, assignment_stats, shuffle_individuals

# seed
np.random.seed(1234)

datadir = "/export/storage_covidvaccine/Data"


# block level data from demest_blocks.py
auxtag = "_111_blk_4q_tract" #config_tag and setting_tag
geog_utils = pd.read_csv(f"{datadir}/Analysis/Demand/agent_utils{auxtag}.csv")
geog_utils.columns.tolist()
abd=geog_utils.abd.values
hpi=geog_utils.hpi_quantile.values.astype(int)
distcoef=geog_utils.distcoef.values
# ensure that the distance coefficient is negative TODO:
assert np.all(distcoef < 0)

geog_data = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
geog_data = geog_data.loc[geog_data.blkid.isin(geog_utils.blkid.values), :]
n_individuals = geog_data.population.values

print("Number of geogs:", geog_data.shape[0]) # 377K
print("Number of individuals:", geog_data.population.sum()) # 39M

# distance matrix from block_dist.py
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})

distdf.shape
distdf.columns.tolist()
# Log distance
distdf['logdist'] = np.log(distdf['dist']) 
# keep blkids in data
distdf = distdf.loc[distdf.blkid.isin(geog_utils.blkid.values), :]
distdf.groupby('blkid').size().describe()

# distances should be sorted ascending within each block
locs = distdf.groupby('blkid').locid.apply(np.array).values
dists = distdf.groupby('blkid').logdist.apply(np.array).values


#=====REFRESH MODULES=====
importlib.reload(sys.modules['vax_entities'])
importlib.reload(sys.modules['assignment_funcs'])
from vax_entities import Individual, Geog
from assignment_funcs import initialize_geogs, compute_ranking, random_fcfs, reset_assignments, assignment_stats, shuffle_individuals
#=========================


# time
time1 = time.time()
geogs = initialize_geogs(
    locs,
    dists,
    abd,
    n_individuals,
    )
print("Time to initialize geogs:", time.time() - time1)



n_locations = len(np.unique(distdf.locid.values))
capacity = 10000





# compute ranking
time1 = time.time()
compute_ranking(geogs, distcoef)
print("Time to compute ranking:", time.time() - time1) 


# # shuffle individuals
# indiv_ordering = shuffle_individuals(geogs)

# # assignment
# time1 = time.time()
# random_fcfs(geogs, n_locations, capacity, indiv_ordering)
# print("Time to assign:", time.time() - time1)


# debugging
tt = 23000
ii = 0
geogs[tt].location_ids
geogs[tt].individuals[ii].nlocs_considered
[geogs[tt].individuals[ii+jj].nlocs_considered for jj in range(10)  ]


geogs[tt].location_ids
geogs[tt].distances
geogs[tt].abd
geogs[tt].individuals
geogs[tt].ab_epsilon



# #===================================================================================================
# # Make inputs for demand estimation
# #===================================================================================================

# import pyblp
# # Iteration and Optimization Configurations 
# iteration_config = pyblp.Iteration(method='lm')
# optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-8}) #TODO: tighten
# poolnum = 32

# # df and agent_data

# nsplits = 4
# splits = np.linspace(0, 1, nsplits+1)

# df_read = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
# df_read['hpi_quantile'] = pd.cut(df_read['hpi'], splits, labels=False, include_lowest=True) + 1
# df_read.columns.tolist()

# geog_utils.columns.tolist()

# agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
# agent_data_read = agent_data_read.loc[agent_data_read['zip'].isin(df_read['zip']), :]
# agent_data_read.columns.tolist()

# # TODO: the logdist in this must now be the new assigned distance

# # distmatrix is part of agent_data
# # weights are a function of assignment

# agent_data_assigned = distdf
# # agent_data_assigned['weights'] = ???



# #===================================================================================================
# # I need a function that just takes in the assignment and returns the new utilities and coefficients

# agent_data = agent_data_read #TODO: update with new assigned distance and weights
# df = df_read #TODO: probably wrong but maybe not

# controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
#     'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
#     'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
#     'medianhomevalue', 'popdensity'] 



# formulation1_str = "1 + prices" 
# for qq in range(1, nsplits):
#     formulation1_str += f' + hpi_quantile{qq}'
# for vv in controls:
#     formulation1_str += " + " + vv

# agent_formulation_str =  '0 +' 
# for qq in range(nsplits):
#     agent_formulation_str += f' + hpi_quantile{qq+1}'

# formulation1 = pyblp.Formulation(formulation1_str)
# formulation2 = pyblp.Formulation('1')
# agent_formulation = pyblp.Formulation(agent_formulation_str)

# # initialize pi
# # TODO: initialize pi with the results from the previous iteration
# print("Agent formulation: ", agent_formulation_str)
# agent_vars = agent_formulation_str.split(' + ')
# agent_vars.remove('0')
# pi_init = 0.01*np.ones((1,len(agent_vars)))

# # Instruments - weighted averages of agent-level variables
# for (ii,vv) in enumerate(agent_vars):
#     print(f"demand_instruments{ii}: {vv}")
#     ivcol = pd.DataFrame({f'demand_instruments{ii}': agent_data.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
#     df = df.merge(ivcol, left_on='market_ids', right_index=True)

# # Solve problem
# problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)
# with pyblp.parallel(poolnum): 
#     results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)


# # utilities TODO: streamline this
# pis = results.pi.flatten()
# pilabs = results.pi_labels

# deltas = results.compute_delta(market_id = df['market_ids'])
# deltas_df = pd.DataFrame({'market_ids': df['market_ids'], 'delta': deltas.flatten()})
# # compute block-level utilities: dot product of agent_vars and pis

# agent_utils = agent_data[['blkid', 'market_ids', 'hpi_quantile', 'logdist']].assign(
#     agent_utility = 0,
#     distcoef = 0
# )

# for (ii,vv) in enumerate(pilabs):
#     coef = pis[ii]
#     if 'dist' in vv:
#         print(f"{vv} is a distance term, omitting from ABD and adding to coefficients instead")
#         if vv=='logdist':
#             deltas_df = deltas_df.assign(distcoef = agent_data[vv])
#         elif vv.startswith('logdistXhpi_quantile'):
#             qq = int(vv[-1])
#             agent_utils.loc[:, 'distcoef'] +=  agent_data[f"hpi_quantile{qq}"] * coef

#     else:
#         print(f"Adding {vv} to agent-level utility")
#         agent_utils.loc[:, 'agent_utility'] += agent_data[vv] * coef

# agent_utils = agent_utils.merge(deltas_df, on='market_ids')
# agent_utils = agent_utils.assign(abd = agent_utils['agent_utility'] + agent_utils['delta'])

# #===================================================================================================

# # create a n_geogs x M matrix of location assignments


# x = np.array([3, 4, 2, 1])
# x[np.argpartition(x, 3)]



