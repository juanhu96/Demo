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
testing = True
raw_capacity = 10000 #capacity per location. lower when testing 
max_rank = 5 #maximum rank to offer

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
distdf = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
distdf = distdf.groupby('blkid').head(max_rank).reset_index(drop=True)
distdf = distdf.loc[distdf.blkid.isin(blocks_tokeep), :]

### Block estimation
block_utils = pd.read_csv(f'{resultdir}Demand/agent_results_{raw_capacity}_200_3q.csv', delimiter = ",") 
block_utils = block_utils.loc[block_utils.blkid.isin(blocks_tokeep), :]

distcoefs = block_utils.distcoef.values
abd = block_utils.abd.values
print(len(distcoefs), len(abd))
####################################################################################


# Create Economy object
print("Start creating economy...")
# create economy
dist_grp = distdf.groupby('blkid')
locs = dist_grp.locid.apply(np.array).values # list of lists of location IDs, corresponding to dists
dists = dist_grp.logdist.apply(np.array).values # list of lists of distances, sorted ascending
geog_pops = block.population.values

# print(locs, dists)
# print(len(locs))
economy = vaxclass.Economy(locs, dists, geog_pops, max_rank=max_rank)
print(economy.n_geogs)

# RUN FIXED POINT
'''
print("Entering fixed point loop...\nTime:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()

agent_results, results = fp.run_fp(
    economy=economy,
    capacity=capacity,
    agent_data_full=agent_data_full,
    cw_pop=block,
    df=df,
    product_formulations=product_formulations,
    agent_formulation=agent_formulation,
    coefsavepath=coefsavepath,
    micro_computation_chunks=1 if max_rank <= 50 else 10
)

print("Done with fixed point loop at time:", round(time.time()-time_entered, 2), "seconds")
sys.stdout.flush()
'''

### TODO: currently random_fcfs assures everyone to their last if the previous full, we want to directly drop these
### TODO: this is not done in random_fcfs but in assignment_stats

# a0 = copy.deepcopy(economy.assignments) # initial assignments?
af.random_fcfs(economy, distcoefs, abd, capacity)
af.assignment_stats(economy, max_rank)
