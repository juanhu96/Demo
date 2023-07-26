# demand estimation with capacity constraints

# 0. Draw a random order for demanders once.
# 1. Estimate the demand model under a given matching function (start with closest facility and no capacity constraints)
# 2. Apply estimated demand to the random order of demanders, except the distance for each demander is based on the closest location with available capacity by the time they get their choice.
# 3. Repeat step 1 assuming the location matching from step 2. Repeat iteratively until you reach a fixed point

# Notation: I use "geog" to denote a block/tract. We're using blocks now.

import pandas as pd
import numpy as np
import pyblp
import importlib

try:
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de


np.random.seed(1234)

datadir = "/export/storage_covidvaccine/Data"

#=================================================================
# Stuff for assignment (run once)
#=================================================================
# block level data from demest_blocks.py
auxtag = "_110_blk_4q_tract" #config_tag and setting_tag
geog_utils = pd.read_csv(f"{datadir}/Analysis/Demand/agent_utils{auxtag}.csv")
geog_utils.columns.tolist() #['blkid', 'market_ids', 'hpi_quantile', 'logdist', 'agent_utility', 'distcoef', 'delta', 'abd']
geog_utils['delta'].value_counts() # 0 for all

# TODO: subsetting blocks to test
testing = True
if testing:
    test_frac = 0.03
    test_ngeog = int(round(test_frac*geog_utils.shape[0], 0))
    blocks_tokeep = np.random.choice(geog_utils.blkid, size=test_ngeog, replace=False)
    geog_utils = geog_utils.loc[geog_utils.blkid.isin(blocks_tokeep), :]
else:
    test_frac = 1
    blocks_tokeep = geog_utils.blkid.values


abd=geog_utils.abd.values
distcoefs=geog_utils.distcoef.values
# ensure that the distance coefficient is negative TODO:
assert np.all(distcoefs < 0)

cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
cw_pop = cw_pop.loc[cw_pop.blkid.isin(blocks_tokeep), :]
n_individuals = cw_pop.population.values

print("Number of geogs:", cw_pop.shape[0]) # 377K
print("Number of individuals:", cw_pop.population.sum()) # 39M

# from block_dist.py. this is in long format
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})

# keep blkids in data
distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]
distdf.groupby('blkid').size().describe()
# nearest for each block
distnearest = distdf.groupby('blkid').head(1)

# distances should be sorted ascending within each block.
dist_grp = distdf.groupby('blkid')
locs = dist_grp.locid.apply(np.array).values
dists = dist_grp.logdist.apply(np.array).values


economy = af.initialize_economy(
    locs=locs,
    dists=dists,
    n_individuals=n_individuals
)

ordering = af.shuffle_individuals(economy.individuals)

locids = np.unique(distdf.locid.values) #TODO: is this the right index/order?
capacity = 10000 * test_frac 

#=================================================================
# Stuff for demand estimation (run once)
#=================================================================

nsplits = 4
splits = np.linspace(0, 1, nsplits+1)
hpi_level = 'tract'

results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results{auxtag}.pkl")
problem0 = results.problem
agent_vars=str(problem0.agent_formulation).split(' + ')
control_vars = str(problem0.product_formulations[0]).split(' + ').remove('1')
df_colstokeep = control_vars + ['market_ids', 'firm_ids']

df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv", usecols=df_colstokeep)
df = df.loc[df.market_ids.isin(geog_utils.market_ids), :]

df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)

# agent_data that has all the locs
agent_data_full = distdf.merge(geog_utils[['blkid', 'market_ids', 'hpi_quantile']], on='blkid', how='left')
agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_bins=False, add_dummies=True, add_dist=True)
agent_data_full = agent_data_full.assign(nodes = 0)



#=================================================================
# Stuff for assignment (run in loop)
#=================================================================
af.compute_pref(economy, abd, distcoefs)

# assignment
af.random_fcfs(economy, locids, capacity, ordering)
#322 seconds TODO: do better


#=================================================================
# Stuff for demand estimation (run in loop)
#=================================================================
agent_data = af.subset_locs(economy.offers, agent_data_full) #subset to locs that were offered, add weights column
df = af.assignment_shares(economy.assignments, cw_pop) #collapse to market-level shares following assignment

df = de.add_ivcols(df, agent_data, agent_vars=agent_vars)

agent_results = de.estimate_demand(df, agent_data, problem0)
abd = agent_results['abd'].values
distcoefs = agent_results['distcoef'].values



#=================================================================
#=====REFRESH MODULES=====

importlib.reload(af)
importlib.reload(de)
try:
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de

#=================================================================


#=================================================================
# debugging

af.pref_stats(economy)
af.assignment_stats(economy)
# TODO: why is percentage offered not 100%? 



