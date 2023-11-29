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
testing = sys.argv == [''] #test if running in terminal, full run if running in shell script

testing = False 

capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 #capacity per location. lower when testing TODO: change to 10000 or sth
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 200 #maximum rank to offer
nsplits = int(sys.argv[3]) if len(sys.argv) > 3 else 3 #number of HPI quantiles
hpi_level = sys.argv[4] if len(sys.argv) > 4 else 'zip' #zip or tract
mnl = True #TODO: switch to False for original model
if mnl:
    max_rank = 5 
    


# in rundemest_assm.sh we have, e.g.:
# nohup python3 /users/facsupport/zhli/VaxDemandDistance/Demand/demest_assm.py 10000 10 > demest_assm_10000_10.out &

only_constant = False #if True, only estimate constant term in demand. default to False
no_dist_heterogeneity = False #if True, no distance heterogeneity in demand. default to False
cap_coefs_to0 = False # if we get positive distance coefficients, set them to 0 TODO:

setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
setting_tag += "_const" if only_constant else ""
setting_tag += "_nodisthet" if no_dist_heterogeneity else ""
setting_tag += "_capcoefs0" if cap_coefs_to0 else ""
setting_tag += f"_{hpi_level}hpi" if hpi_level != 'zip' else ""
setting_tag += "_mnl" if mnl else ""

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
    cw_pop = cw_pop.loc[cw_pop.blkid.isin(blocks_tokeep), :]
    print("********** TESTING **********")
    print(f"Number of geogs: {ngeog}")
    print(f"Number of geogs to keep: {test_ngeog}")
    print(f"Capacity: {capacity}")
else:
    test_frac = 1
    blocks_tokeep = blocks_unique


cw_pop.sort_values(by=['blkid'], inplace=True)

print("Number of geogs:", cw_pop.shape[0]) # 377K
print("Number of individuals:", cw_pop.population.sum()) # 39M
sys.stdout.flush()



#=================================================================
# Data for demand estimation: market-level data, agent-level data
#=================================================================

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
# subset when testing
if testing:
    agent_data_read = agent_data_read.loc[agent_data_read.blkid.isin(blocks_tokeep), :]


# keep markets in both
mkts_in_both = set(df['market_ids'].tolist()).intersection(set(agent_data_read['market_ids'].tolist()))
print("Number of markets:", len(mkts_in_both))
agent_data_read = agent_data_read.loc[agent_data_read.market_ids.isin(mkts_in_both), :]
df = df.loc[df.market_ids.isin(mkts_in_both), :]

# subset distances and crosswalk also
cw_pop = cw_pop.loc[cw_pop.market_ids.isin(mkts_in_both), :]

saved_distdf_subset = False #TODO: change to False if changing sample
if saved_distdf_subset:
    distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist_blockstokeep.csv")
    if testing:
        distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]
else:
    # from block_dist.py. this is in long format. sorted by blkid, then logdist
    distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
    distdf = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)
    distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]
    # keep blkids in data
    distdf = distdf.loc[distdf.blkid.isin(agent_data_read.blkid.unique()), :]
    if not testing:
        distdf.to_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist_blockstokeep.csv", index=False)
        print(f"Saved distdf to {datadir}/Intermediate/ca_blk_pharm_dist_blockstokeep.csv")
    else:
        print("Not saving distdf because testing")
        distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]


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
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank, mnl=mnl)

print("Done creating economy at time:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()


if mnl:
    print("MNL distance distribution by rank:")
    print([np.mean([dd[ii] for dd in dists]) for ii in range(len(dists[0]))])
#=================================================================
#=================================================================

# RUN FIXED POINT

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
    coefsavepath=coefsavepath, #TODO: save intermediate in iter
    micro_computation_chunks=1 if max_rank <= 50 else 10,
    cap_coefs_to0=cap_coefs_to0,
    mnl=mnl,
    verbose=True #TODO: change to False
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

if not testing:
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
print(results)
tablevars = []
for qq in range(1, nsplits+1): #e.g. quantiles 1,2,3
    tablevars += [f'logdistXhpi_quantile{qq}']

for qq in range(1, nsplits): #e.g. quantiles 1,2,3
    tablevars += [f'hpi_quantile{qq}']

tablevars = tablevars + controls + ['1']
print("Table variables:", tablevars)

coefrows, serows, varlabels = de.start_table(tablevars)
coefrows, serows = de.fill_table(results, coefrows, serows, tablevars)

coefrows = [r + "\\\\ \n" for r in coefrows]
serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]

latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
for (ii,vv) in enumerate(varlabels):
    latex += coefrows[ii]
    latex += serows[ii]

latex += "\\bottomrule\n\\end{tabular}\n\n\nNote: $^{\\dag}$ indicates a variable at the block level."
table_path = f"{outdir}/coeftables/coeftable_{setting_tag}.tex" 

with open(table_path, "w") as f:
    print(f"Saved table at: {table_path}")
    f.write(latex)

print("Done!")


# #=================================================================
# #=================================================================
# #=====REFRESH MODULES=====
# import importlib
# import copy

# importlib.reload(vaxclass)
# importlib.reload(af)
# importlib.reload(de)
# importlib.reload(fp)
# try:
#     from demand_utils import vax_entities as vaxclass
#     from demand_utils import assignment_funcs as af
#     from demand_utils import demest_funcs as de
#     from demand_utils import fixed_point as fp
# except:
#     from Demand.demand_utils import vax_entities as vaxclass
#     from Demand.demand_utils import assignment_funcs as af
#     from Demand.demand_utils import demest_funcs as de
#     from Demand.demand_utils import fixed_point as fp

#=================================================================
# #=================================================================
# FOR FIX COMPARISON  


# results = pd.read_pickle(f"{outdir}/results_8000_200_3q.pkl")

# for nsplits in [3, 4]:
#     tablevars = []
#     for qq in range(1, nsplits+1): #e.g. quantiles 1,2,3
#         tablevars += [f'logdistXhpi_quantile{qq}']

#     for qq in range(1, nsplits): #e.g. quantiles 1,2,3
#         tablevars += [f'hpi_quantile{qq}']

#     tablevars = tablevars + controls + ['1']
#     print("Table variables:", tablevars)

#     coefrows, serows, varlabels = de.start_table(tablevars)
#     for capacity in [8000,10000,12000]:
#         setting_tag = f"{capacity}_200_{nsplits}q"
#         results = pd.read_pickle(f"{outdir}/results_{setting_tag}.pkl")
#         print(setting_tag)
#         print(results)
#         coefrows, serows = de.fill_table(results, coefrows, serows, tablevars)

#     coefrows = [r + "\\\\ \n" for r in coefrows]
#     serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]



#     latex = "\\begin{tabular}{lccc}\n \\toprule\n\\midrule\n \\ Variable & \\multicolumn{3}{c}{Capacity} \\\\ \n & 8,000 & 10,000 & 12,000 \\\\ \n"
#     for (ii,vv) in enumerate(varlabels):
#         latex += coefrows[ii]
#         latex += serows[ii]
#     latex += "\\bottomrule\n\\end{tabular}\n\n"

#     table_path = f"{outdir}/coeftables/coeftable_{nsplits}q.tex"

#     with open(table_path, "w") as f:
#         print(f"Saved table at: {table_path}")
#         f.write(latex)

