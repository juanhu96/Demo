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


#=================================================================
# SETTINGS
#=================================================================
testing = False # TODO: subsetting blocks to test
capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 #capacity per location. lower when testing
max_rank = 10 #maximum rank to offer


#=================================================================
# Data for assignment: distances, block-market crosswalk, population
#=================================================================

print(f"Testing: {testing}")
print(f"Capacity: {capacity}")

cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
blocks_unique = np.unique(cw_pop.blkid.values)
markets_unique = np.unique(cw_pop.market_ids.values)
if testing:
    test_frac = 0.05
    ngeog = len(blocks_unique)
    test_ngeog = int(round(test_frac*ngeog, 0))
    blocks_tokeep = np.random.choice(blocks_unique, size=test_ngeog, replace=False)
    capacity = capacity * test_frac  #capacity per location. lower when testing
else:
    test_frac = 1
    blocks_tokeep = blocks_unique


cw_pop = cw_pop.loc[cw_pop.blkid.isin(blocks_tokeep), :]
cw_pop.sort_values(by=['blkid'], inplace=True)

print("Number of geogs:", cw_pop.shape[0]) # 377K
print("Number of individuals:", cw_pop.population.sum()) # 39M

# from block_dist.py. this is in long format. sorted by blkid, then logdist
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
# assert (distdf['blkid'].is_monotonic_increasing and distdf.groupby('blkid')['logdist'].apply(lambda x: x.is_monotonic_increasing).all())
# keep blkids in data
distdf = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)
distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]


#=================================================================
# Data for demand estimation: market-level data, agent-level data
#=================================================================

nsplits = 4
hpi_level = 'tract'
ref_lastq = False


controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
    'collegegrad', 'unemployment', 'poverty', 'logmedianhhincome', 
    'logmedianhomevalue', 'logpopdensity']
df_colstokeep = controls + ['market_ids', 'firm_ids', 'shares', 'prices'] + ['hpi']

formulation1_str = "1 + prices"
for qq in range(1, nsplits):
    formulation1_str += f' + hpi_quantile{qq}'
for vv in controls:
    formulation1_str += " + " + vv

agent_formulation_str = '0 +'
for qq in range(1, nsplits):
    agent_formulation_str += f' logdistXhpi_quantile{qq} +'
if ref_lastq:
    agent_formulation_str += ' logdist'
else:
    agent_formulation_str += f' logdistXhpi_quantile{nsplits}'

formulation1 = pyblp.Formulation(formulation1_str)
formulation2 = pyblp.Formulation('1')
product_formulations = (formulation1, formulation2)
agent_formulation = pyblp.Formulation(agent_formulation_str)


# read in product_data
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df = df.loc[df.market_ids.isin(markets_unique), :]
df = df.loc[:, df_colstokeep]
df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)


# read in agent_data
agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=['blkid', 'market_ids'])

# keep markets in both
mkts_in_both = set(df['market_ids'].tolist()).intersection(set(agent_data_read['market_ids'].tolist()))
print("Number of markets:", len(mkts_in_both))
agent_data_read = agent_data_read.loc[agent_data_read.market_ids.isin(mkts_in_both), :]
df = df.loc[df.market_ids.isin(mkts_in_both), :]

# subset distances and crosswalk also
distdf = distdf.loc[distdf.blkid.isin(agent_data_read.blkid.unique()), :]
cw_pop = cw_pop.loc[cw_pop.market_ids.isin(mkts_in_both), :]


# add HPI quantile to agent data
if hpi_level == 'zip':
    agent_data_read = agent_data_read.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')
elif hpi_level == 'tract':
    tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") #from prep_tracts.py
    blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) #from block_cw.py
    splits = np.linspace(0, 1, nsplits+1)
    agent_data_read = agent_data_read.merge(blk_tract_cw, on='blkid', how='left')
    tract_hpi['hpi_quantile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1
    agent_data_read = agent_data_read.merge(tract_hpi[['tract', 'hpi_quantile']], on='tract', how='left')

# merge distances
agent_data_full = distdf.merge(agent_data_read[['blkid', 'market_ids', 'hpi_quantile']], on='blkid', how='left')
agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_bins=False, add_dummies=True, add_dist=True)
agent_data_full['nodes'] = 0
# merge in market population
mktpop = cw_pop.groupby('market_ids').agg({'population': 'sum'}).reset_index()
agent_data_full = agent_data_full.merge(mktpop, on='market_ids', how='left')

print("agent_data_full.shape:", agent_data_full.shape)


# Create Economy object
print("Start creating economy at time:", round(time.time()-time_entered, 2), "seconds")
# create economy
dist_grp = distdf.groupby('blkid')
locs = dist_grp.locid.apply(np.array).values
dists = dist_grp.logdist.apply(np.array).values
geog_pops = cw_pop.population.values
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank)

print("Done creating economy at time:", round(time.time()-time_entered, 2), "seconds")

#=================================================================
#=================================================================
# things modified in FP:
# economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank)


#=================================================================
#=================================================================

# RUN FIXED POINT

print("Entering fixed point loop...\nTime:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

fp.run_fp(
    economy=economy,
    capacity=capacity,
    agent_data_full=agent_data_full,
    cw_pop=cw_pop,
    df=df,
    product_formulations=product_formulations,
    agent_formulation=agent_formulation
)


# #=================================================================
# #=================================================================
# #=================================================================
# #=====REFRESH MODULES=====
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

# # #=================================================================
# # # debugging
# tol = 1e-3
# problem = problem_init
# gtol = 1e-8
# poolnum = 1
# capacity = 10000
# max_rank = 10
# iter = 0
# converged = False
# a0 = copy.deepcopy(economy.assignments)
# pi_init = results.pi


# cw_pop = cw_pop.reset_index(drop=True)
# cw_pop['geog_ind'] = cw_pop.index
# mktinds = np.unique(cw_pop['market_ids'])
# wdists = np.zeros(len(mktinds)) #preallocate
# mm_where = [np.where(cw_pop['market_ids'].values == mm)[0] for mm in mktinds]
# mm_where[9]
# dists_mm = [np.concatenate([economy.dists[tt] for tt in mm]) for mm in mm_where]

# sorted_indices = [np.argsort(dd) for dd in dists_mm]
# dists_mm_sorted = [dists_mm[i][sorted_indices[i]] for i in range(len(dists_mm))]



# # assignment
# af.random_fcfs(economy, distcoefs, abd, capacity)
# af.assignment_stats(economy)

# converged = fp.wdist_checker(a0, economy, cw_pop, tol) if iter > 0 else False

# # demand estimation
# # subset agent_data to the locations that were actually offered
# offer_weights = np.concatenate(economy.offers)
# offer_inds = np.flatnonzero(offer_weights)
# offer_weights = offer_weights[offer_inds]
# print(f"Number of agents: {len(offer_inds)}")
# agent_data = agent_data_full.loc[offer_inds].copy()
# agent_data['weights'] = offer_weights/agent_data['population']

# # df = af.assignment_shares(df, economy.assignments, cw_pop) 
# pi_iter = pi_init if iter == 0 else results.pi
# results, agent_results = de.estimate_demand(df, agent_data, problem, pi_init=pi_iter, gtol=gtol, poolnum=poolnum)
# abd = agent_results['abd'].values
# distcoefs = agent_results['distcoef'].values

# print(f"\n************\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}")

# iter += 1
# a0 = copy.deepcopy(economy.assignments)
# af.random_fcfs(economy, distcoefs, abd, capacity)

# a1 = copy.deepcopy(a0)
# a1[9]
# a1[9][1] = 150
# a1[9][1] = 150
# a1[9]
# fp.wdist_checker(dists_mm_sorted, sorted_indices, a0, a1, wdists, tol)

# a0[9]
# economy.assignments[9]
# wdists[9]
# pi_iter
# results.pi

# # #=================================================================

# # # debugging
# # TODO: compute wasserstein distance between original and new assignments at some point in fp. 
# # figure out why it's not converging
# # output both convergence metrics

# a0 = copy.deepcopy(economy.assignments)
# af.random_fcfs(economy, distcoefs, abd, capacity)
# if af.assignment_stats(economy) < 0.999:
#     print("Warning: not all individuals are offered")



# # #=================================================================
# from itertools import compress
# economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank)
# distcoefs = [-0.1 for ii in range(economy.n_geogs)]
# abd = [1 for ii in range(economy.n_geogs)]

# af.random_fcfs(economy, distcoefs, abd, capacity)
# af.assignment_stats(economy)

# a0 = copy.deepcopy(economy.assignments)
# original_assm = a0
# distcoefs = [-0.101 for ii in range(economy.n_geogs)]
# af.random_fcfs(economy, distcoefs, abd, capacity)
# af.assignment_stats(economy)
# # converged = fp.wdist_checker(a0, economy, cw_pop, 0.001)

# wdists = []
# for mm in np.unique(cw_pop.market_ids.values):
# # mm = 90001
#     mm_ind = cw_pop['market_ids'].values == mm
#     dists_mm = list(compress(economy.dists, mm_ind))
#     original_assm_mm = list(compress(original_assm, mm_ind))
#     new_assm_mm = list(compress(economy.assignments, mm_ind))

#     dists0, p0 = fp.probdist_distance(original_assm_mm, dists_mm)
#     dists1, p1 = fp.probdist_distance(new_assm_mm, dists_mm)
#     wasserstein_dist01 = fp.wasserstein_distance(p0, p1, dists0)
#     wdists.append(wasserstein_dist01)

# pd.Series(wdists).describe()
# sum([dd > 0 for dd in wdists])

# np.dot(dists0, p0)
# np.dot(dists1, p1)

# # fp.wdist_checker(economy.assignments, economy, cw_pop, tol=1e-3)



# # #=================================================================
# # outdir = "/export/storage_covidvaccine/Data/Analysis/Demand"
# # gtol=1e-8
# # poolnum=1
# # capacity = 10000
# # max_rank = 10
# # 



# # # plug in coefs


# # # # read previous demand estimation results
# # # results = pyblp.read_pickle(f"{outdir}/pyblp_results_fp.pkl")
# # # results.to_pickle(f"{outdir}/pyblp_results_fp_freeze.pkl")
# # # agent_results = pd.read_csv(f"{outdir}/agent_results_fp.csv")
# # # agent_results.to_csv(f"{outdir}/agent_results_fp_freeze.csv")
# # results = pyblp.read_pickle(f"{outdir}/pyblp_results_fp_freeze.pkl")
# # agent_data = pd.read_csv(f"{outdir}/agent_results_fp_freeze.csv")
# # results.pi

# # # assignment
# # distcoefs = agent_data.distcoef.values
# # abd = agent_data.abd.values
# # af.random_fcfs(economy, distcoefs, abd, capacity)
# # assm0 = copy.deepcopy(economy.assignments)

# # # demand estimation
# # offer_weights = np.concatenate(economy.offers)
# # offer_inds = np.flatnonzero(offer_weights)
# # offer_weights = offer_weights[offer_inds]
# # print(f"Number of agents: {len(offer_inds)}")
# # agent_data = agent_data_full.loc[offer_inds].copy()
# # agent_data['weights'] = offer_weights/agent_data['population']
# # df = af.assignment_shares(df, economy.assignments, cw_pop)
# # pi_iter = results.pi
# # results, agent_data = de.estimate_demand(df, agent_data, problem_init, pi_init=pi_iter, gtol=gtol, poolnum=poolnum)
# # results.pi

# # # assignment
# # distcoefs = agent_data.distcoef.values
# # abd = agent_data.abd.values
# # af.random_fcfs(economy, distcoefs, abd, capacity)
# # assm1 = copy.deepcopy(economy.assignments)

# # fp.assignment_difference(assm0, assm1, economy.total_pop)
# # diff_ind = [np.any(a0!=a1) for (a0, a1) in zip(assm0, assm1)]
# # np.mean(diff_ind)

# # dists = economy.dists[diff_ind]

# # # first differences
# # fd = [[dd[ii] - dd[ii-1] for ii in range(1, max_rank)] for dd in dists]
# # meanfd = np.mean(np.mean(fd, axis=1))
# # print(meanfd)






# # # #=================================================================



# # # # economy = vaxclass.Economy(locs, dists, geog_pops)

# # # capacity = 10000 * test_frac  #capacity per location. lower when testing
# # # af.random_fcfs(economy, distcoefs, abd, capacity)
# # # af.assignment_stats(economy, max_rank=15)

# # # offers = economy.offers
# # # maxgeogoffers = np.array([np.max(np.flatnonzero(offers[tt])) for tt in range(economy.n_geogs)])
# # # hioffer = np.where(maxgeogoffers>75)
# # # cw_pop.iloc[hioffer]
# # # hizips = np.unique(cw_pop.iloc[hioffer].market_ids.values)
# # # hizips
# # # df.loc[df.market_ids.isin(hizips), 'shares']
# # # df.columns.tolist()
# # # import copy
# # # df_orig = copy.deepcopy(df)
# # # df = af.assignment_shares(df, economy.assignments, cw_pop)
# # # df.loc[df.market_ids.isin(hizips), 'shares']

# # # occs = np.array(list(economy.occupancies.values()))
# # # np.mean(occs==12000)


# # # # subset agent_data to the locations that were actually offered
# # # offer_weights = np.concatenate(economy.offers)
# # # offer_weights.shape
# # # offer_inds = np.flatnonzero(offer_weights)
# # # offer_inds.shape
# # # agent_data = agent_data_full.loc[offer_inds]
# # # agent_data['weights'] = offer_weights[offer_inds]/agent_data['population']

# # # agent_data.shape
# # # agent_data.blkid.nunique()
# # # problem0 = problem_init

# # # agent_formulation = problem0.agent_formulation
# # # agent_vars = str(agent_formulation).split(' + ')

# # # df = de.add_ivcols(df, agent_data, agent_vars=agent_vars)
# # # problem = pyblp.Problem(
# # #     product_formulations=problem0.product_formulations, 
# # #     product_data=df, 
# # #     agent_formulation=agent_formulation, 
# # #     agent_data=agent_data)


# # # gtol = 1e-8
# # # optimization_config = pyblp.Optimization('trust-constr', {'gtol':gtol})


# # # pi_init = results.pi
# # # poolnum = 32
# # # iteration_config = pyblp.Iteration(method='lm')
# # # with pyblp.parallel(poolnum): 
# # #     results = problem.solve(
# # #         pi=pi_init,
# # #         sigma = 0, 
# # #         iteration = iteration_config, 
# # #         optimization = optimization_config)



# # # agent_data.blkid.nunique()


# # # fp.run_fp(
# # #     economy=economy,
# # #     abd=abd,
# # #     distcoefs=distcoefs,
# # #     capacity=10000000,
# # #     agent_data_full=agent_data_full,
# # #     cw_pop=cw_pop,
# # #     df=df,
# # #     problem=problem_init,
# # #     gtol=1e-8,
# # #     pi_init=results.pi
# # # )


# # # # # #=================================================================
# # # # # # Stuff for assignment (run in loop)
# # # # # #=================================================================
# # # # a0 = economy.assignments

# # # # af.random_fcfs(economy, distcoefs, abd, capacity) 
# # # # a1 = economy.assignments
# # # # adiff = fp.assignment_difference(a0, a1, economy.total_pop)
# # # # a0[100]
# # # # a1[100]
# # # # adiff

# # # # offer_weights = np.concatenate(economy.offers)
# # # # offer_inds = np.flatnonzero(offer_weights)
# # # # agent_data = agent_data_full.loc[offer_inds].copy()
# # # # agent_data['weights'] = offer_weights[offer_inds]

# # # # df = af.assignment_shares(df, economy.assignments, cw_pop)

# # # # pi_init, agent_results = de.estimate_demand(df, agent_data, problem_init, pi_init=results.pi)

# # # # # #=================================================================
# # # # # #=================================================================
# # # # # debugging

# # # # # assignment
# # # # af.pref_stats(economy)

# # # # df = af.assignment_shares(df, economy.assignments, cw_pop)

# # # # af.assignment_stats(economy) 




# # # # #=================================================================
# # # # # debugging
# # # # af.assignment_stats(economy) #
# # # # a1 = economy.assignments
# # # # af.random_fcfs(economy, capacity+15)
# # # # a2 = economy.assignments

# # # # tt = 300
# # # # ii = 0

# # # # geo1 = [a1[tt]]
# # # # geo2 = [a2[tt]]
# # # # geo1
# # # # geo2
# # # # fp.assignment_difference(geo1, geo2)

# # # # fp.assignment_difference(a1, a2)


# # # # #=================================================================
# # # # # Stuff for demand estimation (run in loop)
# # # # #=================================================================
# # # # agent_data = af.subset_locs(economy.offers, agent_data_full) #subset to locs that were offered, add weights column

# # # # df = af.assignment_shares(df, economy.assignments, cw_pop)#collapse to market-level shares following assignment


# # # # pi_result, agent_results = de.estimate_demand(df, agent_data, problem_init)
# # # # abd = agent_results['abd'].values
# # # # distcoefs = agent_results['distcoef'].values



# # # # #=================================================================
# # # # # debugging
# # # # #=================================================================

# # # # # gumbel_variance = np.pi**2 / 6
# # # # # for epsilon_opt in ["zero", "logistic"]:
# # # # #     for scale in [1, gumbel_variance, 1/gumbel_variance]:
# # # # #         print(f"\n**********\nepsilon_opt: {epsilon_opt}, scale: {scale}")
# # # # # # ...
# # # # #         af.pref_stats(economy) # match with empirical shares?

