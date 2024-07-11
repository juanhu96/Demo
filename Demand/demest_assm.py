# demand estimation with capacity constraints

# 0. Draw a random order for demanders once.
# 1. Estimate the demand model under a given matching function (start with closest facility and no capacity constraints)
# 2. Apply estimated demand to the random order of demanders, except the distance for each demander is based on the closest location with available capacity by the time they get their choice.
# 3. Repeat step 1 assuming the location matching from step 2. Repeat iteratively until you reach a fixed point

# Notation: I use "geog" to denote a block/tract. We're using blocks now.

import sys
import os
import pandas as pd
import numpy as np
import pyblp
import time

print("Entering demest_assm.py at time:", time.strftime("%Y-%m-%d %H:%M:%S"))

#=================================================================
# SETTINGS
#=================================================================

capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 #capacity per location.
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 5 #maximum rank to offer
nsplits = int(sys.argv[3]) if len(sys.argv) > 3 else 4 #number of HPI quantiles
hpi_level = 'zip' #zip or tract
mnl = any([arg == 'mnl' for arg in sys.argv]) # False for original model
distbins = any(['distbin' in arg for arg in sys.argv]) # False for original model
flexible_consideration = any(['flex' in arg for arg in sys.argv]) # False for original model
# 10 miles for rural i.e. popdensity_group = 3, 2 miles for popdensity_group = 2, 1 mile for popdensity_group = 1
flex_thresh = dict(zip(["urban", "suburban", "rural"], [2,3,15])) #thresholds (in km) for flexible consideration
distbin_cuts = [1,5]
n_distbins = len(distbin_cuts) + 1
logdist_above = any(['logdistabove' in arg for arg in sys.argv]) # False for original model. Argument will be e.g. "logdistabove1" to have log(dist) for >1km and flat for <=1km. 
if logdist_above:
    logdist_above_arg = [arg for arg in sys.argv if 'logdistabove' in arg][0]
    logdist_above_thresh = float(logdist_above_arg.replace('logdistabove', ''))

levelshift = any(['levelshift' in arg for arg in sys.argv]) # False for original model. Whether to have a level shift for logdistabove

only_constant = any(['const' in arg for arg in sys.argv]) #if True, only estimate constant term in demand. default to False
no_dist_heterogeneity = any(['nodisthet' in arg for arg in sys.argv]) #if True, no distance heterogeneity in demand. default to False
cap_coefs_to0 = any(['capcoef' in arg for arg in sys.argv]) #if True, set coefficients on distance to 0 when capacity is 0. default to False

strict_capacity = any(['strict_capacity' in arg for arg in sys.argv]) #if True, only allow assignment to locations with capacity. default to False

dummy_location = any(['dummy_location' in arg for arg in sys.argv]) #if True, assign capacity-violators a dummy location that's very far. default to False. Argument will be e.g. "dummy_location10" to have dummy location at 10km.
if dummy_location:
    dummy_location_arg = [arg for arg in sys.argv if 'dummy_location' in arg][0]
    dummy_location_dist = float(dummy_location_arg.replace('dummy_location', ''))


# in rundemest_assm.sh we have, e.g.:
# nohup python3 /users/facsupport/zhli/VaxDemandDistance/Demand/demest_assm.py 12000 1 4 "flex" &


setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
setting_tag += "_const" if only_constant else ""
setting_tag += "_nodisthet" if no_dist_heterogeneity else ""
setting_tag += "_capcoefs0" if cap_coefs_to0 else ""
setting_tag += f"_{hpi_level}hpi" if hpi_level != 'zip' else ""
setting_tag += f"_distbins_at{str(distbin_cuts).replace(', ', '_').replace('[', '').replace(']', '')}" if distbins else ""
setting_tag += "_mnl" if mnl else ""
setting_tag += "_flex" if flexible_consideration else ""
setting_tag += f"thresh{str(list(flex_thresh.values())).replace(', ', '_').replace('[', '').replace(']', '')}" if flexible_consideration else ""
setting_tag += f"_logdistabove{logdist_above_thresh}" if logdist_above else ""
setting_tag += "_levelshift" if levelshift else ""
setting_tag += "_strict" if strict_capacity else ""
setting_tag += "_dummyloc" if dummy_location else ""
setting_tag += f"{dummy_location_dist}" if dummy_location else ""

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result/Demand"

datestr = time.strftime("%Y%m%d-%H%M")
if len(sys.argv) > 1:
    if not os.path.exists(f"{outdir}/logs/{setting_tag}"):
        os.makedirs(f"{outdir}/logs/{setting_tag}")

    # Redirect stdout and stderr to a log file only when additional command line arguments are present
    datestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = open(f"{outdir}/logs/{setting_tag}/{datestr}.log", "w")
    sys.stdout = log_file
    sys.stderr = log_file


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


randomseed = 1234
np.random.seed(randomseed)

coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs"

#=================================================================
# Data for assignment: distances, block-market crosswalk, population
#=================================================================

print(f"Setting tag: {setting_tag}")
print(f"Capacity: {capacity}")
print(f"Max rank: {max_rank}")
print(f"Number of HPI quantiles: {nsplits}")
print(f"MNL: {mnl}")
print(f"Flexible consideration: {flexible_consideration}")
if flexible_consideration:
    print(f"Flexible consideration thresholds: {flex_thresh}")
if logdist_above:
    print(f"Logdist from: {logdist_above_thresh}")
print(f"HPI level: {hpi_level}")
print(f"Distance bins: {distbins}")
if distbins:
    print(f"Distance bin cuts: {distbin_cuts}")
    print(f"Number of distance bins: {n_distbins}")
print(f"Only constant: {only_constant}")
print(f"No distance heterogeneity: {no_dist_heterogeneity}")
print(f"Cap coefs to 0: {cap_coefs_to0}")
print(f"np.random.seed: {randomseed}")
print(f"Coef save path: {coefsavepath}")
print(f"Strict capacity: {strict_capacity}")
print(f"Dummy location: {dummy_location}")
if dummy_location:
    print(f"Dummy location distance: {dummy_location_dist}")

#=================================================================
# Data for demand estimation: market-level data, agent-level data
#=================================================================

ref_lastq = False

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
    'collegegrad', 'unemployment', 'poverty', 'logmedianhhincome', 
    'logmedianhomevalue', 'logpopdensity']
df_colstokeep = controls + ['market_ids', 'firm_ids', 'shares', 'prices'] + ['hpi'] + ['popdensity_group']

formulation1_str = "1 + prices"
for qq in range(1, nsplits):
    formulation1_str += f' + hpi_quantile{qq}'
for vv in controls:
    formulation1_str += " + " + vv

if only_constant:
    formulation1_str = "1 + prices"

assert no_dist_heterogeneity + distbins + logdist_above <= 1
if no_dist_heterogeneity:
    agent_formulation_str = '0 + logdist'
elif distbins:
    agent_formulation_str = '0 +'
    for qq in range(1, nsplits+1):
        for dd in range(1, n_distbins):
            agent_formulation_str += f' distbin{dd}Xhpi_quantile{qq} +'
    agent_formulation_str = agent_formulation_str[:-1] # remove the last +
elif logdist_above: #for each HPI quantile, logdist for >1km + level shift for >1km
    agent_formulation_str = '0 +'
    for qq in range(1, nsplits+1):
        agent_formulation_str += f' logdist_abovethreshXhpi_quantile{qq} +' 
        if levelshift:
            agent_formulation_str += f' abovethreshXhpi_quantile{qq} +'
    agent_formulation_str = agent_formulation_str[:-1] # remove the last +
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
df = df.loc[:, df_colstokeep]
df = de.hpi_dist_terms(df, nsplits=nsplits, add_hpi_bins=True, add_hpi_dummies=True, add_dist=False)

# read in agent_data
agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv", usecols=['blkid', 'market_ids'])

# read in crosswalk with population
cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/cw_pop.csv")
# total population
print("Total population:", np.sum(cw_pop.population.values))

# distdf is from block_dist.py. this is in long format. sorted by blkid, then logdist
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})

# keep blkids in data
distdf = distdf.loc[distdf.blkid.isin(agent_data_read.blkid.unique()), :]

# consideration set:
# we want distdf_withinthresh to include those within 1/2/10 miles based on whether df['popdensity_group'] is 3/2/1)
if flexible_consideration:
    if max_rank > 10:
        print("WARNING: flexible_consideration is True but max_rank > 10. Setting max_rank to 10.")
        max_rank = 10
        setting_tag = "maxrankwarn_" + setting_tag
    # grab zips from cw_pop
    distdf = distdf.merge(cw_pop[['market_ids', 'blkid']], on='blkid', how='left')
    # grab popdensity_group from df
    distdf = distdf.merge(df[['market_ids', 'popdensity_group']], on='market_ids', how='left')

    distdf_maxrank = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)
    distdf_list = [distdf_maxrank]
    assert set(distdf.popdensity_group) == set(flex_thresh.keys())
    for grp in flex_thresh.keys():
        # grab distdf for this popdensity_group and within the threshold
        distdf_grp = distdf.loc[(distdf.popdensity_group == grp) & (distdf.logdist <= np.log(flex_thresh[grp])), :]
        distdf_list.append(distdf_grp)
        
    # take the union
    distdf = pd.concat(distdf_list).drop_duplicates().reset_index(drop=True)
    # sort by blkid, then logdist
    distdf.sort_values(by=['blkid', 'logdist'], inplace=True)

else:
    distdf = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)


# check if distdf is sorted by blkid
assert np.all(distdf.blkid.values == np.sort(distdf.blkid.values))

distdf = distdf[['blkid', 'locid', 'logdist', 'dist']]

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

if distbins:
    agent_data_full = de.hpi_dist_terms(agent_data_full, dist_varname = 'dist', nsplits=nsplits, add_hpi_bins=False, add_hpi_dummies=True, add_dist=False, add_distbins = True, distbin_cuts=distbin_cuts)
else:
    agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_hpi_bins=False, add_hpi_dummies=True, add_dist=True)

if logdist_above:
    agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_hpi_bins=False, add_hpi_dummies=True, add_dist=False, logdist_above=logdist_above, logdist_above_thresh=logdist_above_thresh)


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
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank, mnl=mnl)

print("Done creating economy at time:", round(time.time()-time_entered, 2), "seconds")

# RUN FIXED POINT


if strict_capacity:
    coefloadpath = coefsavepath.replace("_strict", "")
    pi_init = np.load(coefloadpath + ".npy")

print("Entering fixed point loop...\nTime:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

agent_results, results, agent_loc_data = fp.run_fp(
    economy=economy,
    capacity=capacity,
    agent_data_full=agent_data_full,
    cw_pop=cw_pop,
    df=df,
    product_formulations=product_formulations,
    agent_formulation=agent_formulation,
    coefsavepath=coefsavepath, 
    micro_computation_chunks=1 if max_rank <= 50 else 10,
    cap_coefs_to0=cap_coefs_to0,
    mnl=mnl,
    dampener=0.5,
    verbose=True,
    setting_tag=setting_tag,
    outdir=outdir,
    strict_capacity=strict_capacity,
    dummy_location=dummy_location,
    dummy_location_dist=dummy_location_dist if dummy_location else None,
    pi_init = pi_init if strict_capacity else None
)


# TESTING=========================================================

# totpop = np.sum(cw_pop.population.values)
# df_pops = cw_pop.groupby('market_ids').agg({'population': 'sum'}).reset_index()
# df = df.merge(df_pops, on='market_ids', how='left')
# vaxpop = np.sum(df.population.values * df.shares.values)
# print("Total population:", totpop)
# print("Vaccinated population:", vaxpop)
# print("Vaccinated population share:", vaxpop/totpop)
# # vaccinated population matches 73.64%, which is the assigned population in random FCFS
#=================================================================

print("Done with fixed point loop at time:", round(time.time()-time_entered, 2), "seconds")

sys.stdout.flush()

# save agent_results
try:
    agent_results[['blkid', 'hpi_quantile', 'market_ids', 'abd', 'distcoef']].to_csv(f"{outdir}/agent_results_{setting_tag}.csv", index=False)
    print(f"Saved agent_results to {outdir}/agent_results_{setting_tag}.csv")
    results.to_pickle(f"{outdir}/results_{setting_tag}.pkl")
    print(f"Saved results to {outdir}/results_{setting_tag}.pkl")
    agent_loc_data.to_csv(f"{outdir}/agent_loc_data_{setting_tag}.csv", index=False)
except: #if no storage space 
    agent_results[['blkid', 'market_ids', 'abd', 'distcoef']].to_csv(f"/export/storage_adgandhi/MiscLi/agent_results_{setting_tag}.csv", index=False)
    print(f"Saved agent_results to /export/storage_adgandhi/MiscLi/agent_results_{setting_tag}.csv")
    results.to_pickle(f"/export/storage_adgandhi/MiscLi/results_{setting_tag}.pkl")
    print(f"Saved results to /export/storage_adgandhi/MiscLi/results_{setting_tag}.pkl")
    agent_loc_data.to_csv(f"/export/storage_adgandhi/MiscLi/agent_loc_data_{setting_tag}.csv", index=False)



# #=================================================================
# #=================================================================
# Write coefficient table
results = pd.read_pickle(f"{outdir}/results_{setting_tag}.pkl")
table_path = f"{outdir}/coeftables/coeftable_{setting_tag}.tex" 
de.write_table(results, table_path)

print("Done!")
print("Total time in minutes:", (time.time()-time_entered)/60)


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


if len(sys.argv) > 1:
    # Close log file only when additional command line arguments are present
    log_file.close()
