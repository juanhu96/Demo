# this is just a script for speed testing.

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
outdir = "/export/storage_covidvaccine/Result/Demand"


#=================================================================
# SETTINGS
#=================================================================
# testing = sys.argv == [''] #test if running in terminal, full run if running in shell script
testing = False  #TODO:
capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 #capacity per location. 
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 200 #maximum rank to offer
nsplits = int(sys.argv[3]) if len(sys.argv) > 3 else 3 #number of HPI quantiles

# in rundemest_assm.sh we have, e.g.:
# nohup python3 /users/facsupport/zhli/VaxDemandDistance/Demand/demest_assm.py 10000 10 > demest_assm_10000_10.out &

only_constant = False #if True, only estimate constant term in demand. default to False
no_dist_heterogeneity = False #if True, no distance heterogeneity in demand. default to False

setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
setting_tag += "_const" if only_constant else ""
setting_tag += "_nodisthet" if no_dist_heterogeneity else ""

print(setting_tag)
coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs" if not testing else None


#=================================================================
# Data for assignment: distances, block-market crosswalk, population
#=================================================================

print(f"Testing: {testing}")
print(f"Capacity: {capacity}")
print(f"Max rank: {max_rank}")
print(f"Number of HPI quantiles: {nsplits}")
print(f"Setting tag: {setting_tag}")
print(f"Coef save path: {coefsavepath}")
sys.stdout.flush()

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
sys.stdout.flush()

# from block_dist.py. this is in long format. sorted by blkid, then logdist
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
# assert (distdf['blkid'].is_monotonic_increasing and distdf.groupby('blkid')['logdist'].apply(lambda x: x.is_monotonic_increasing).all())
# keep blkids in data
distdf = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)
distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]





#=================================================================
# Data for demand estimation: market-level data, agent-level data
#=================================================================

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

if only_constant:
    formulation1_str = "1 + prices"

if no_dist_heterogeneity:
    agent_formulation_str = '0 + logdist'
else:
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
sys.stdout.flush()


# read in agent_data
agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=['blkid', 'market_ids'])
agent_data_read = agent_data_read.loc[agent_data_read.blkid.isin(blocks_tokeep), :]

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
agent_data_full['nodes'] = 0 #for pyblp
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
# round to integers 
geog_pops = [np.round(ll).astype(int) for ll in geog_pops]
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank)

print("Done creating economy at time:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

# agent_results, results, agent_loc_data = fp.run_fp(
#     economy=economy,
#     capacity=capacity,
#     agent_data_full=agent_data_full,
#     cw_pop=cw_pop,
#     df=df,
#     product_formulations=product_formulations,
#     agent_formulation=agent_formulation,
#     coefsavepath=coefsavepath, #TODO: save intermediate in iter
#     micro_computation_chunks=1 if max_rank <= 50 else 10,
#     verbose=testing
# )
converged = False
iter = 0
dists_mm_sorted, sorted_indices, wdists = fp.wdist_init(cw_pop, economy.dists)
agent_unique_data = agent_data_full.drop_duplicates(subset=['blkid']).copy()


len(agent_unique_data.market_ids.unique())

# IN LOOP

offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
assert len(offer_weights) == agent_data_full.shape[0]
offer_inds = np.flatnonzero(offer_weights)
offer_weights = offer_weights[offer_inds]
print(f"Number of agents: {len(offer_inds)}")
agent_loc_data = agent_data_full.loc[offer_inds].copy()
agent_loc_data['weights'] = offer_weights/agent_loc_data['population']

pi_init = 0.001*np.ones((1, len(str(agent_formulation).split('+')))) #initialize pi to last result, unless first iteration
results = de.estimate_demand(df, agent_loc_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=1e-8, poolnum=1, verbose=False)

coefs = results.pi.flatten()
agent_results = de.compute_abd(results, df, agent_unique_data, coefs=coefs)

abd = agent_results['abd'].values
distcoefs = agent_results['distcoef'].values

print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")

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


# import cProfile

# cProfile.run('af.random_fcfs( economy=economy, distcoefs=distcoefs, abd=abd, capacity=capacity)')

af.random_fcfs(
    economy=economy,
    distcoefs=distcoefs,
    abd=abd,
    capacity=capacity
)



loctt = economy.locs[0]
locid = 3112

loctt = np.delete(loctt, np.where(loctt == locid)[0])
