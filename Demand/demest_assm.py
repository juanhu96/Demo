# demand estimation with capacity constraints

# 0. Draw a random order for demanders once.
# 1. Estimate the demand model under a given matching function (start with closest facility and no capacity constraints)
# 2. Apply estimated demand to the random order of demanders, except the distance for each demander is based on the closest location with available capacity by the time they get their choice.
# 3. Repeat step 1 assuming the location matching from step 2. Repeat iteratively until you reach a fixed point

# Notation: I use "geog" to denote a block/tract. We're using blocks now.

import pandas as pd
import numpy as np
import pyblp
import sys
import time

print("Entering demest_assm.py")
time_entered = time.time()


try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
    from demand_utils import fixed_point as fp
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de
    from Demand.demand_utils import fixed_point as fp

np.random.seed(1234)

datadir = "/export/storage_covidvaccine/Data"
auxtag = "_111_blk_4q_tract" #config_tag and setting_tag

#=================================================================
# Stuff for assignment (run once)
#=================================================================
# block level data from demest_blocks.py
geog_utils = pd.read_csv(f"{datadir}/Analysis/Demand/agent_utils{auxtag}.csv") #output of demest_blocks.py
geog_utils.columns.tolist() #['blkid', 'market_ids', 'hpi_quantile', 'logdist', 'agent_utility', 'distcoef', 'delta', 'abd']


testing = True   # TODO: subsetting blocks to test
print(f"Testing: {testing}")
print(f"Setting tag: {auxtag}")
if testing:
    test_frac = 0.05
    test_ngeog = int(round(test_frac*geog_utils.shape[0], 0))
    blocks_tokeep = np.random.choice(geog_utils.blkid, size=test_ngeog, replace=False)
    geog_utils = geog_utils.loc[geog_utils.blkid.isin(blocks_tokeep), :]
else:
    test_frac = 1
    blocks_tokeep = geog_utils.blkid.values

geog_utils.sort_values(by=['blkid'], inplace=True)
abd=geog_utils.abd.values
distcoefs=geog_utils.distcoef.values

cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
cw_pop = cw_pop.loc[cw_pop.blkid.isin(blocks_tokeep), :]
cw_pop.sort_values(by=['blkid'], inplace=True)

geog_pops = cw_pop.population.values

print("Number of geogs:", cw_pop.shape[0]) # 377K
print("Number of individuals:", cw_pop.population.sum()) # 39M
sys.stdout.flush()

# from block_dist.py. this is in long format. sorted by blkid, then logdist
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
# assert (distdf['blkid'].is_monotonic_increasing and distdf.groupby('blkid')['logdist'].apply(lambda x: x.is_monotonic_increasing).all())


# keep blkids in data
distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]
dist_grp = distdf.groupby('blkid')
locs = dist_grp.locid.apply(np.array).values
dists = dist_grp.logdist.apply(np.array).values

print("Start creating economy at time:", round(time.time()-time_entered, 2), "seconds")
# create economy
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=10)

print("Done creating economy at time:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

capacity = 10000 * test_frac  #capacity per location. lower when testing
max
#=================================================================
# Stuff for demand estimation (run once)
#=================================================================

nsplits = 4
hpi_level = 'tract'

results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results{auxtag}.pkl")
problem_init = results.problem
agent_vars=str(problem_init.agent_formulation).split(' + ')
control_vars = str(problem_init.product_formulations[0]).split(' + ')
control_vars.remove('1')
df_colstokeep = control_vars + ['market_ids', 'firm_ids', 'shares']

df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df = df.loc[df.market_ids.isin(geog_utils.market_ids), :]
df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)
df = df.loc[:, df_colstokeep]

# agent_data that has all the locs
agent_data_full = distdf.merge(geog_utils[['blkid', 'market_ids', 'hpi_quantile']], on='blkid', how='left')
agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_bins=False, add_dummies=True, add_dist=True)
agent_data_full['nodes'] = 0
# merge in market population
mktpop = cw_pop.groupby('market_ids').agg({'population': 'sum'}).reset_index()
agent_data_full = agent_data_full.merge(mktpop, on='market_ids', how='left')


#=================================================================
#=================================================================
# things modified in FP:
# economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=10)
# df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
# df = df.loc[df.market_ids.isin(geog_utils.market_ids), :]
# df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)
# df = df.loc[:, df_colstokeep]
#=================================================================
#=================================================================

# #=================================================================
# #=================================================================
# RUN FIXED POINT

print("Entering fixed point loop...\nTime:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

fp.run_fp(
    economy=economy,
    abd=abd,
    distcoefs=distcoefs,
    capacity=capacity,
    agent_data_full=agent_data_full,
    cw_pop=cw_pop,
    df=df,
    problem=problem_init,
    gtol=1e-8,
    poolnum=1,
    micro_computation_chunks = 8,
    pi_init=results.pi
)


#=================================================================
#=================================================================
#=================================================================
#=====REFRESH MODULES=====
import importlib
import copy

importlib.reload(vaxclass)
importlib.reload(af)
importlib.reload(de)
importlib.reload(fp)
try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
    from demand_utils import fixed_point as fp
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de
    from Demand.demand_utils import fixed_point as fp

# #=================================================================
# # debugging
# outdir = "/export/storage_covidvaccine/Data/Analysis/Demand"
# gtol=1e-8
# poolnum=1
# capacity = 10000
# max_rank = 10
# agent_data_full = agent_data_full.groupby('blkid').head(max_rank).reset_index(drop=True)



# # plug in coefs


# # # read previous demand estimation results
# # results = pyblp.read_pickle(f"{outdir}/pyblp_results_fp.pkl")
# # results.to_pickle(f"{outdir}/pyblp_results_fp_freeze.pkl")
# # agent_results = pd.read_csv(f"{outdir}/agent_results_fp.csv")
# # agent_results.to_csv(f"{outdir}/agent_results_fp_freeze.csv")
# results = pyblp.read_pickle(f"{outdir}/pyblp_results_fp_freeze.pkl")
# agent_data = pd.read_csv(f"{outdir}/agent_results_fp_freeze.csv")
# results.pi

# # assignment
# distcoefs = agent_data.distcoef.values
# abd = agent_data.abd.values
# af.random_fcfs(economy, distcoefs, abd, capacity)
# assm0 = copy.deepcopy(economy.assignments)

# # demand estimation
# offer_weights = np.concatenate(economy.offers)
# offer_inds = np.flatnonzero(offer_weights)
# offer_weights = offer_weights[offer_inds]
# print(f"Number of agents: {len(offer_inds)}")
# agent_data = agent_data_full.loc[offer_inds].copy()
# agent_data['weights'] = offer_weights/agent_data['population']
# df = af.assignment_shares(df, economy.assignments, cw_pop)
# pi_iter = results.pi
# results, agent_data = de.estimate_demand(df, agent_data, problem_init, pi_init=pi_iter, gtol=gtol, poolnum=poolnum)
# results.pi

# # assignment
# distcoefs = agent_data.distcoef.values
# abd = agent_data.abd.values
# af.random_fcfs(economy, distcoefs, abd, capacity)
# assm1 = copy.deepcopy(economy.assignments)

# fp.assignment_difference(assm0, assm1, economy.total_pop)
# diff_ind = [np.any(a0!=a1) for (a0, a1) in zip(assm0, assm1)]
# np.mean(diff_ind)

# dists = economy.dists[diff_ind]

# # first differences
# fd = [[dd[ii] - dd[ii-1] for ii in range(1, max_rank)] for dd in dists]
# meanfd = np.mean(np.mean(fd, axis=1))
# print(meanfd)






# # #=================================================================



# # # economy = vaxclass.Economy(locs, dists, geog_pops)

# # capacity = 10000 * test_frac  #capacity per location. lower when testing
# # af.random_fcfs(economy, distcoefs, abd, capacity)
# # af.assignment_stats(economy, max_rank=15)

# # offers = economy.offers
# # maxgeogoffers = np.array([np.max(np.flatnonzero(offers[tt])) for tt in range(economy.n_geogs)])
# # hioffer = np.where(maxgeogoffers>75)
# # cw_pop.iloc[hioffer]
# # hizips = np.unique(cw_pop.iloc[hioffer].market_ids.values)
# # hizips
# # df.loc[df.market_ids.isin(hizips), 'shares']
# # df.columns.tolist()
# # import copy
# # df_orig = copy.deepcopy(df)
# # df = af.assignment_shares(df, economy.assignments, cw_pop)
# # df.loc[df.market_ids.isin(hizips), 'shares']

# # occs = np.array(list(economy.occupancies.values()))
# # np.mean(occs==12000)


# # # subset agent_data to the locations that were actually offered
# # offer_weights = np.concatenate(economy.offers)
# # offer_weights.shape
# # offer_inds = np.flatnonzero(offer_weights)
# # offer_inds.shape
# # agent_data = agent_data_full.loc[offer_inds]
# # agent_data['weights'] = offer_weights[offer_inds]/agent_data['population']

# # agent_data.shape
# # agent_data.blkid.nunique()
# # problem0 = problem_init

# # agent_formulation = problem0.agent_formulation
# # agent_vars = str(agent_formulation).split(' + ')

# # df = de.add_ivcols(df, agent_data, agent_vars=agent_vars)
# # problem = pyblp.Problem(
# #     product_formulations=problem0.product_formulations, 
# #     product_data=df, 
# #     agent_formulation=agent_formulation, 
# #     agent_data=agent_data)


# # gtol = 1e-8
# # optimization_config = pyblp.Optimization('trust-constr', {'gtol':gtol})


# # pi_init = results.pi
# # poolnum = 32
# # iteration_config = pyblp.Iteration(method='lm')
# # with pyblp.parallel(poolnum): 
# #     results = problem.solve(
# #         pi=pi_init,
# #         sigma = 0, 
# #         iteration = iteration_config, 
# #         optimization = optimization_config)



# # agent_data.blkid.nunique()


# # fp.run_fp(
# #     economy=economy,
# #     abd=abd,
# #     distcoefs=distcoefs,
# #     capacity=10000000,
# #     agent_data_full=agent_data_full,
# #     cw_pop=cw_pop,
# #     df=df,
# #     problem=problem_init,
# #     gtol=1e-8,
# #     pi_init=results.pi
# # )


# # # # #=================================================================
# # # # # Stuff for assignment (run in loop)
# # # # #=================================================================
# # # a0 = economy.assignments

# # # af.random_fcfs(economy, distcoefs, abd, capacity) 
# # # a1 = economy.assignments
# # # adiff = fp.assignment_difference(a0, a1, economy.total_pop)
# # # a0[100]
# # # a1[100]
# # # adiff

# # # offer_weights = np.concatenate(economy.offers)
# # # offer_inds = np.flatnonzero(offer_weights)
# # # agent_data = agent_data_full.loc[offer_inds].copy()
# # # agent_data['weights'] = offer_weights[offer_inds]

# # # df = af.assignment_shares(df, economy.assignments, cw_pop)

# # # pi_init, agent_results = de.estimate_demand(df, agent_data, problem_init, pi_init=results.pi)

# # # # #=================================================================
# # # # #=================================================================
# # # # debugging

# # # # assignment
# # # af.pref_stats(economy)

# # # df = af.assignment_shares(df, economy.assignments, cw_pop)

# # # af.assignment_stats(economy) 




# # # #=================================================================
# # # # debugging
# # # af.assignment_stats(economy) #
# # # a1 = economy.assignments
# # # af.random_fcfs(economy, capacity+15)
# # # a2 = economy.assignments

# # # tt = 300
# # # ii = 0

# # # geo1 = [a1[tt]]
# # # geo2 = [a2[tt]]
# # # geo1
# # # geo2
# # # fp.assignment_difference(geo1, geo2)

# # # fp.assignment_difference(a1, a2)


# # # #=================================================================
# # # # Stuff for demand estimation (run in loop)
# # # #=================================================================
# # # agent_data = af.subset_locs(economy.offers, agent_data_full) #subset to locs that were offered, add weights column

# # # df = af.assignment_shares(df, economy.assignments, cw_pop)#collapse to market-level shares following assignment


# # # pi_result, agent_results = de.estimate_demand(df, agent_data, problem_init)
# # # abd = agent_results['abd'].values
# # # distcoefs = agent_results['distcoef'].values



# # # #=================================================================
# # # # debugging
# # # #=================================================================

# # # # gumbel_variance = np.pi**2 / 6
# # # # for epsilon_opt in ["zero", "logistic"]:
# # # #     for scale in [1, gumbel_variance, 1/gumbel_variance]:
# # # #         print(f"\n**********\nepsilon_opt: {epsilon_opt}, scale: {scale}")
# # # # # ...
# # # #         af.pref_stats(economy) # match with empirical shares?

