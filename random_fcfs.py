#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug, 2023
@Author: Jingyuan Hu 

Temp file for random order fcfs
"""

import pyblp
import sys
import time
import pandas as pd
import numpy as np

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

datadir='/export/storage_covidvaccine/Data/'
resultdir='/export/storage_covidvaccine/Result/'
testing = False
raw_capacity = 10000 # capacity per location. lower when testing 
max_rank = 5 # max rank to offer
nsplits = 3 # 3 quartiles

expdirpath = '/export/storage_covidvaccine/Result/MaxVaxHPIDistBLP/M5_K10000/Dollar/'
opt_constr = 'assigned'
chain_name = 'Dollar'


####################################################################################
### Block basic info
block = pd.read_csv(f'{datadir}/Analysis/Demand/block_data.csv', usecols=["blkid", "market_ids", "population"]) 
blocks_unique = np.unique(block.blkid.values)
markets_unique = np.unique(block.market_ids.values)

if testing:
    test_frac = 0.0005
    ngeog = len(blocks_unique)
    test_ngeog = int(round(test_frac*ngeog, 0))
    np.random.seed(1234)
    blocks_tokeep = np.random.choice(blocks_unique, size=test_ngeog, replace=False)
    capacity = raw_capacity * test_frac  #capacity per location. lower when testing (because only a few blocks competing for sites)
else:
    test_frac = 1
    blocks_tokeep = blocks_unique
    capacity = raw_capacity

block = block.loc[block.blkid.isin(blocks_tokeep), :]
block.sort_values(by=['blkid'], inplace=True)

### Distance pairs
# distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
distdf = pd.read_csv(f'{expdirpath}{opt_constr}/ca_blk_{chain_name}_dist_total.csv', dtype={'locid': int, 'blkid': int})
# distdf = pd.read_csv(f'{datadir}Intermediate/ca_blk_{chain_name}_dist.csv', dtype={'locid': int, 'blkid': int}) 

distdf = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)
distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]


### Block estimation
block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{raw_capacity}_200_3q.csv', delimiter = ",") 
block_utils = block_utils.loc[block_utils.blkid.isin(blocks_tokeep), :]

### Keep markets in both
df = pd.read_csv(f"{datadir}Analysis/Demand/demest_data.csv")
df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=False, add_dist=False)
df = df.loc[df.market_ids.isin(markets_unique), :]
mkts_in_both = set(df['market_ids'].tolist()).intersection(set(block['market_ids'].tolist()))
print("Number of markets:", len(mkts_in_both))

### Subset blocks and add HPI
block = block.loc[block.market_ids.isin(mkts_in_both), :]
df = df.loc[df.market_ids.isin(mkts_in_both), :]
distdf = distdf.loc[distdf.blkid.isin(block.blkid.unique()), :]
block = block.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')

'''
### Estimation results
distcoefs = block_utils.distcoef.values
abd = block_utils.abd.values

####################################################################################
dist_grp = distdf.groupby('blkid')
locs = dist_grp.locid.apply(np.array).values # list of lists of location IDs, corresponding to dists
dists = dist_grp.logdist.apply(np.array).values # list of lists of distances, sorted ascending
geog_pops = block.population.values


economy = vaxclass.Economy(locs, dists, geog_pops, max_rank)
af.random_fcfs_eval(economy, distcoefs, abd, capacity)
af.assignment_stats_eval(economy, max_rank)

# np.savetxt(f'{resultdir}locs_{raw_capacity}.csv', np.stack(locs, axis=0), fmt='%s')
# np.savetxt(f'{resultdir}dists_{raw_capacity}.csv', np.stack(dists, axis=0))
# np.savetxt(f'{resultdir}assignment_{raw_capacity}.csv', assignment_mat, fmt='%s')

np.savetxt(f'{expdirpath}{opt_constr}/locs_{raw_capacity}_{chain_name}.csv', np.stack(locs, axis=0), fmt='%s')
np.savetxt(f'{expdirpath}{opt_constr}/dists_{raw_capacity}_{chain_name}.csv', np.stack(dists, axis=0))
np.savetxt(f'{expdirpath}{opt_constr}/assignment_{raw_capacity}_{chain_name}.csv',  np.array(economy.assignments), fmt='%s')
'''
####################################################################################
### Report results

# TODO: compute vaccination rates at block-level now
### Vaccination rates by quartile


# find the HPI quartile of each block
locs = np.genfromtxt(f'{expdirpath}{opt_constr}/locs_{raw_capacity}_{chain_name}.csv', delimiter = "")
dists = np.genfromtxt(f'{expdirpath}{opt_constr}/dists_{raw_capacity}_{chain_name}.csv', delimiter = "")
assignment = np.genfromtxt(f'{expdirpath}{opt_constr}/assignment_{raw_capacity}_{chain_name}.csv', delimiter = "")


for quantile in range(1, nsplits+1):
    
    print(quantile)

    block_quantile = block[block.hpi_quantile == quantile]
    locs_quantile = locs[block.hpi_quantile == quantile]
    dists_quantile = dists[block.hpi_quantile == quantile]
    assignment_quantile = assignment[block.hpi_quantile == quantile]

    total_pop_quantile = sum(block_quantile.population)
    total_assignment_quantile = np.round(np.sum(assignment_quantile) / 1000000, 2)
    rate_quantile = np.round(total_assignment_quantile / total_pop_quantile * 100, 2)

    print(total_assignment_quantile, rate_quantile)

    
