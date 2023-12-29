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
import copy
from matplotlib import pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW


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
testing = sys.argv == [''] #test if running in terminal, full run if running in shell script
testing = False  #TODO:
capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 #capacity per location. lower when testing
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 50 #maximum rank to offer
nsplits = int(sys.argv[3]) if len(sys.argv) > 3 else 3 #number of HPI quantiles

# in rundemest_assm.sh we have, e.g.:
# nohup python3 /users/facsupport/zhli/VaxDemandDistance/Demand/demest_assm.py 10000 10 > demest_assm_10000_10.out &

setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
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
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank)

print("Done creating economy at time:", round(time.time()-time_entered, 2), "seconds")

#=================================================================
#=================================================================
#=================================================================
#TESTING ASSIGNMENT

len(economy.locs[0])
economy.locs[0]

len(economy.dists[0])
economy.dists[0]

geog_pops[0]
len(economy.abepsilon)
len(economy.abepsilon[0])
economy.abepsilon[0]

len(economy.epsilon_diff)
len(economy.epsilon_diff[0])
economy.epsilon_diff[0]


len(economy.offers[0])
len(economy.assignments[0])

gtol= 1e-10
poolnum = 1
micro_computation_chunks= 1
tol = 0.005


converged = False
iter = 0
dists_mm_sorted, sorted_indices, wdists = fp.wdist_init(cw_pop, economy.dists)


#=================================================================

offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
offer_inds = np.flatnonzero(offer_weights)
offer_weights = offer_weights[offer_inds]
print(f"Number of agents: {len(offer_inds)}")
agent_data = agent_data_full.loc[offer_inds].copy()
agent_data['weights'] = offer_weights/agent_data['population']


pi_init = results.pi if iter > 0 else 0.001*np.ones((1, len(str(agent_formulation).split('+')))) #initialize pi to last result, unless first iteration
results, agent_results = de.estimate_demand(df, agent_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=gtol, poolnum=poolnum, verbose=False)
abd = agent_results['abd'].values
distcoefs = agent_results['distcoef'].values
print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")


a0 = copy.deepcopy(economy.assignments)

##### ASSIGNMENT #####

economy.abepsilon = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)] 

# reset occupancies 
economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
full_locations = set()
# reset offers and assignments
economy.offers = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]
economy.assignments = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]


economy.ordering[0]
tt, ii = economy.ordering[0]

jj=0
ll = economy.locs[tt][jj]
ll 
economy.occupancies[ll]
ll in full_locations


economy.offers[tt]
len(economy.offers[tt])
economy.offers[tt][jj]
len(economy.assignments[tt])
economy.locs[tt]
len(economy.locs[tt])
economy.dists[tt]
len(economy.dists[tt])
economy.abepsilon[tt]
len(economy.abepsilon[tt])
economy.epsilon_diff[tt]
len(economy.epsilon_diff[tt])
geog_pops[tt]

####
#### RESET:
economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
full_locations = set()
# reset offers and assignments
economy.offers = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]
economy.assignments = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]
####
####


for (tt,ii) in economy.ordering:
    for (jj,ll) in enumerate(economy.locs[tt]): #locs[tt] is ordered by distance from geography tt, in ascending order
        print(f"tt: {tt}, ii: {ii}, jj: {jj}, ll: {ll}")
        if ll not in full_locations or jj==len(economy.locs[tt])-1:
            print(f"   OFFERED")
            # -> the individual is offered here
            economy.offers[tt][jj] += 1
            if economy.abepsilon[tt][jj] > economy.epsilon_diff[tt][ii]: # -> the individual is vaccinated here
                print(f"   ASSIGNED")
                economy.assignments[tt][jj] += 1
                economy.occupancies[ll] += 1
                if economy.occupancies[ll] == capacity:
                    full_locations.add(ll)
            break #TODO: check if we need one more break or something

tt = 1
economy.epsilon_diff[tt]
np.mean(economy.epsilon_diff[tt])
economy.epsilon_diff[tt][25]
economy.abepsilon[tt][0]
economy.abepsilon[tt]




#=================================================================
af.random_fcfs(economy, distcoefs, abd, capacity)


#=================================================================

# TODO: look at why assignment/offer plot is wrong - it's not wrong, just the denominator is the total assigned vs total population
qtlticks = np.linspace(0, 1, 101)


def weighted_quantiles(data, weights, quantiles):
    # Sort data and rearrange weights
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute the cumulative sum of weights
    cum_weights = np.cumsum(sorted_weights)
    
    # Total sum of weights
    total_weight = np.sum(weights)

    # Compute quantiles
    quantile_positions = quantiles * total_weight
    quantile_values = np.interp(quantile_positions, cum_weights, sorted_data)

    return quantile_values

offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
offer_inds = np.flatnonzero(offer_weights)
offer_weights = offer_weights[offer_inds]

economy.assignments[0]
len(economy.assignments)
assm_weights = np.concatenate(economy.assignments) 
len(np.concatenate(economy.assignments))
assm_inds = np.flatnonzero(assm_weights)
assm_weights = assm_weights[assm_inds]
dists_assigned = np.concatenate(economy.dists)[assm_inds]
assm_qtls = np.exp(weighted_quantiles(dists_assigned, assm_weights, qtlticks))




ranks = [range(len(x)) for x in economy.dists]

len(np.concatenate(ranks))
economy.n_geogs
len(np.concatenate(ranks))/economy.n_geogs

len(offer_inds)

ranks_offered = np.concatenate(ranks)[offer_inds]
ranks_assigned = np.concatenate(ranks)[assm_inds]
ranks_offered_qtls = weighted_quantiles(ranks_offered, offer_weights, qtlticks)
ranks_assigned_qtls = weighted_quantiles(ranks_assigned, assm_weights, qtlticks)




ranks_offered_qtls_dsw =  DescrStatsW(data=ranks_offered, weights=offer_weights).quantile(probs=qtlticks)
ranks_offered_qtls_dsw.iloc[70:89]


ranks_assigned_qtls_dsw =  DescrStatsW(data=ranks_assigned, weights=assm_weights).quantile(probs=qtlticks)
ranks_assigned_qtls_dsw.iloc[70:89]


# plt.clf()
# plt.plot(qtlticks, np.quantile(ranks_offered, qtlticks))
# plt.savefig('quantiles_ranks_offered_unw.png')

# plt.clf()
# plt.plot(qtlticks, np.quantile(ranks_assigned, qtlticks))
# plt.savefig('quantiles_ranks_assigned_unw.png')

# line plot
# ranks offered

plt.clf()
plt.plot(qtlticks, ranks_offered_qtls)
plt.xlabel('Quantile')
plt.ylabel('Rank Offered')
plt.savefig('quantiles_ranks_offered_full.png')

# ranks assigned
plt.clf()
plt.plot(qtlticks, ranks_assigned_qtls)
plt.xlabel('Quantile')
plt.ylabel('Rank Assigned')
plt.savefig('quantiles_ranks_assigned_full.png')


ranks_fortotal = np.concatenate(ranks)
len(ranks_fortotal)
len(offer_inds)
len(assm_inds)
offered_unassigned = np.setdiff1d(offer_inds, assm_inds)
ranks_fortotal[offered_unassigned] = 10000
ranks_fortotal = ranks_fortotal[offer_inds]
len(offer_weights)
len(ranks_fortotal)
np.sum(offer_weights==0)
np.sum(ranks_fortotal==10000)
np.sum(ranks_fortotal==10000)/len(ranks_fortotal)
ranks_fortotal_qtls = DescrStatsW(data=ranks_fortotal, weights=offer_weights).quantile(probs=qtlticks)
ranks_fortotal_qtls
np.mean(ranks_fortotal)
plt.clf()
plt.plot(qtlticks, ranks_fortotal_qtls)
plt.xlabel('Quantile')
plt.ylabel('Rank Assigned')
plt.savefig('quantiles_ranks_assigned_totaldenom.png')


#=================================================================




len(distcoefs)
len(abd)
len(economy.locs)
len(np.concatenate(economy.locs))
len(np.concatenate(economy.locs))/50


a0[0]
economy.assignments[0]

af.assignment_stats(economy)

converged = fp.wdist_checker(a0, economy.assignments, dists_mm_sorted, sorted_indices, wdists, tol)

### back to start of while loop

offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
len(np.concatenate(economy.offers))
offer_inds = np.flatnonzero(offer_weights)
len(offer_inds)
offer_weights = offer_weights[offer_inds]
len(offer_weights)
print(f"Number of agents: {len(offer_inds)}")
agent_data = agent_data_full.loc[offer_inds].copy()
agent_data['weights'] = offer_weights/agent_data['population']

pi_init = results.pi
results, agent_results = de.estimate_demand(df, agent_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=gtol, poolnum=poolnum, verbose=False)
abd = agent_results['abd'].values
distcoefs = agent_results['distcoef'].values

print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")

a0 = copy.deepcopy(economy.assignments)
o0 = copy.deepcopy(economy.offers)
af.assignment_stats(economy)

af.random_fcfs(economy, distcoefs, abd, capacity)
a1 = copy.deepcopy(economy.assignments)
o1 = copy.deepcopy(economy.offers)
af.assignment_stats(economy)

fp.wdist_checker(a0, a1, dists_mm_sorted, sorted_indices, wdists, tol)

maxwdist_ind = np.argmax(wdists)
maxwdist_ind 
len(dists_mm_sorted)
dists_mm_sorted[maxwdist_ind]
len(sorted_indices)
sorted_indices[maxwdist_ind]
len(dists_mm_sorted[maxwdist_ind])
sorted_indices[maxwdist_ind]
len(sorted_indices[maxwdist_ind])

fp.assignment_difference(a0, economy)


len(offer_weights)
agent_data = agent_data_full.loc[offer_inds].copy()
agent_data['weights'] = offer_weights/agent_data['population']

agent_data_full
len(np.concatenate(economy.offers))
#=================================================================
# TODO: DISTANCE VS RANK
logdistsbyrank = [np.mean([economy.dists[tt][rr] for tt in range(economy.n_geogs) if len(economy.dists[tt])>rr]) for rr in range(50)]
leveldistsbyrank = [np.mean([np.exp(economy.dists[tt][rr]) for tt in range(economy.n_geogs) if len(economy.dists[tt])>rr]) for rr in range(50)]

plt.clf()
plt.plot(logdistsbyrank)
plt.xlabel("Rank")
plt.ylabel("Mean log distance")
plt.savefig("logdistsbyrank.png")

plt.clf()
plt.plot(leveldistsbyrank)
plt.xlabel("Rank")
plt.ylabel("Mean distance (km)")
plt.savefig("leveldistsbyrank.png")


#=================================================================
#=================================================================

#=================================================================
#=================================================================
# TESTING

# gtol= 1e-10
# poolnum = 1
# micro_computation_chunks= 1
# tol = 0.005


# converged = False
# iter = 0
# dists_mm_sorted, sorted_indices, wdists = fp.wdist_init(cw_pop, economy.dists)


# #=================================================================
# # WHILE LOOP STARTS

# offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
# offer_inds = np.flatnonzero(offer_weights)
# offer_weights = offer_weights[offer_inds]
# print(f"Number of agents: {len(offer_inds)}")
# agent_data = agent_data_full.loc[offer_inds].copy()
# agent_data['weights'] = offer_weights/agent_data['population']


# pi_init = results.pi if iter > 0 else 0.001*np.ones((1, len(str(agent_formulation).split('+')))) #initialize pi to last result, unless first iteration
# results, agent_results = de.estimate_demand(df, agent_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=gtol, poolnum=poolnum, verbose=False)
# abd = agent_results['abd'].values
# distcoefs = agent_results['distcoef'].values
# print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")


# a0 = copy.deepcopy(economy.assignments)
# af.random_fcfs(economy, distcoefs, abd, capacity)
# af.assignment_stats(economy)
# converged = fp.wdist_checker(a0, economy.assignments, dists_mm_sorted, sorted_indices, wdists, tol)
# print(f"Iteration {iter} complete.\n\n")
# sys.stdout.flush()
# iter += 1



# #=================================================================NEXT ITERATION

# offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
# offer_inds = np.flatnonzero(offer_weights)
# offer_weights = offer_weights[offer_inds]
# print(f"Number of agents: {len(offer_inds)}")
# agent_data = agent_data_full.loc[offer_inds].copy()
# agent_data['weights'] = offer_weights/agent_data['population']


# pi_init = results.pi if iter > 0 else 0.001*np.ones((1, len(str(agent_formulation).split('+')))) #initialize pi to last result, unless first iteration
# results, agent_results = de.estimate_demand(df, agent_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=gtol, poolnum=poolnum, verbose=False)
# abd = agent_results['abd'].values
# distcoefs = agent_results['distcoef'].values
# print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")



# a0 = copy.deepcopy(economy.assignments)
# af.random_fcfs(economy, distcoefs, abd, capacity)
# af.assignment_stats(economy)
# converged = fp.wdist_checker(a0, economy.assignments, dists_mm_sorted, sorted_indices, wdists, tol)
# print(f"Iteration {iter} complete.\n\n")
# sys.stdout.flush()
# iter += 1












#CHECK WDIST COMPUTATION





# WHILE LOOP ENDS
#=================================================================


#=================================================================
#=================================================================
#=================================================================

# RUN FIXED POINT

print("Entering fixed point loop...\nTime:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

agent_results = fp.run_fp(
    economy=economy,
    capacity=capacity,
    agent_data_full=agent_data_full,
    cw_pop=cw_pop,
    df=df,
    product_formulations=product_formulations,
    agent_formulation=agent_formulation,
    coefsavepath=coefsavepath,
    micro_computation_chunks=1 if max_rank <= 50 else 10,
    maxiter = 10 #TODO: remove!!!
)

print("Done with fixed point loop at time:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

# save agent_results

# TODO: add back:

# if not testing:
#     try:
#         agent_results[['blkid', 'hpi_quantile', 'market_ids', 'abd', 'distcoef']].to_csv(f"{outdir}/agent_results_{setting_tag}.csv", index=False)
#         print(f"Saved agent_results to {outdir}/agent_results_{setting_tag}.csv")
#     except: #if no storage space 
#         agent_results[['blkid', 'market_ids', 'abd', 'distcoef']].to_csv(f"/export/storage_adgandhi/MiscLi/agent_results_{setting_tag}.csv", index=False)
#         print(f"Saved agent_results to /export/storage_adgandhi/MiscLi/agent_results_{setting_tag}.csv")

# # #=================================================================
# #=================================================================


#CHECK ASSIGNMENT
# economy.assignments[0]
# economy.assignments[1]
# economy.dists[1]
# economy.occupancies
# quantiles of economy.occupancies
occs = np.array(list(economy.occupancies.values()))
occs
for q in np.linspace(0, 1, 21):
    print(f"{int(q*100)}: {int(np.quantile(occs, q))}")

# histogram
import matplotlib.pyplot as plt

plt.clf()
plt.hist(occs, bins=25)
plt.xlabel('Occupancy')
plt.ylabel('Number of locations')
plt.savefig('occupancies.png')


np.mean(occs>=10000)



def weighted_quantiles(data, weights, quantiles):
    # Sort data and rearrange weights
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute the cumulative sum of weights
    cum_weights = np.cumsum(sorted_weights)
    
    # Total sum of weights
    total_weight = np.sum(weights)

    # Compute quantiles
    quantile_positions = quantiles * total_weight
    quantile_values = np.interp(quantile_positions, cum_weights, sorted_data)

    return quantile_values



# distances offered

offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
offer_inds = np.flatnonzero(offer_weights)
offer_weights = offer_weights[offer_inds]

len(offer_weights)
dists_offered = np.concatenate(economy.dists)[offer_inds]
len(dists_offered)


qtlticks = np.linspace(0, 1, 101)
offer_qtls = np.exp(weighted_quantiles(dists_offered, offer_weights, qtlticks))
# line plot
plt.clf()
plt.plot(qtlticks[:-1], offer_qtls[:-1])
plt.xlabel('Quantile')
plt.ylabel('Distance Offered (km)')
plt.savefig('dist_quantiles.png')

plt.clf()
plt.plot(qtlticks, offer_qtls)
plt.xlabel('Quantile')
plt.ylabel('Distance Offered (km)')
plt.savefig('dist_quantiles_full.png')


# distances assigned
assm_weights = np.concatenate(economy.assignments) 
assm_inds = np.flatnonzero(assm_weights)
assm_weights = assm_weights[assm_inds]
dists_assigned = np.concatenate(economy.dists)[assm_inds]
assm_qtls = np.exp(weighted_quantiles(dists_assigned, assm_weights, qtlticks))
# line plot
plt.clf()
plt.plot(qtlticks[:-1], assm_qtls[:-1])
plt.xlabel('Quantile')
plt.ylabel('Distance Assigned (km)')
plt.savefig('dist_quantiles_assigned.png')

plt.clf()
plt.plot(qtlticks, assm_qtls)
plt.xlabel('Quantile')
plt.ylabel('Distance Assigned (km)')
plt.savefig('dist_quantiles_assigned_full.png')


# ranks offered
ranks = [range(len(x)) for x in economy.dists]
ranks_offered = np.concatenate(ranks)[offer_inds]
ranks_assigned = np.concatenate(ranks)[assm_inds]
ranks_offered_qtls = weighted_quantiles(ranks_offered, offer_weights, qtlticks)
ranks_assigned_qtls = weighted_quantiles(ranks_assigned, assm_weights, qtlticks)


# line plot
# ranks offered

plt.clf()
plt.plot(qtlticks, ranks_offered_qtls)
plt.xlabel('Quantile')
plt.ylabel('Rank Offered')
plt.savefig('quantiles_ranks_offered_full.png')

# ranks assigned
plt.clf()
plt.plot(qtlticks, ranks_assigned_qtls)
plt.xlabel('Quantile')
plt.ylabel('Rank Assigned')
plt.savefig('quantiles_ranks_assigned_full.png')




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
