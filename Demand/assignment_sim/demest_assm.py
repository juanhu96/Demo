# demand estimation with capacity constraints

# 0. Draw a random order for demanders once.
# 1. Estimate the demand model under a given matching function (start with closest facility and no capacity constraints)
# 2. Apply estimated demand to the random order of demanders, except the distance for each demander is based on the closest location with available capacity by the time they get their choice.
# 3. Repeat step 1 assuming the location matching from step 2. Repeat iteratively until you reach a fixed point


# Notation: I use "geog" to denote a block/tract, and "individual" to denote a person within the block/tract

import pandas as pd
import numpy as np

import sys
sys.path.append("/mnt/staff/zhli/VaxDemandDistance/Demand/assignment_sim/") #TODO: only need this when running in terminal

from vax_entities import Individual, Geog, Location
from assignment_funcs import initialize, compute_ranking, random_fcfs, sequential, reset_assignments, assignment_stats

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

geog_data = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
geog_data = geog_data.loc[geog_data.blkid.isin(geog_utils.blkid.values), :]

print("Number of geogs:", geog_data.shape[0]) # 377K
print("Number of individuals:", geog_data.population.sum()) # 39M

# distance matrix from block_dist.py
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist_10.csv")
distdf.columns.tolist()
# keep blkids in data
distdf = distdf.loc[distdf.blkid.isin(geog_data.blkid.values), :]

distmatrix = distdf[[f"km_to_nid{i+1}" for i in range(10)]].values
distmatrix = np.log(distmatrix)
locmatrix = distdf[[f"nid{i+1}" for i in range(10)]].values


# TODO: M: maximum number of locations to consider (for now, just use all 10 we have distances for)
M = distmatrix.shape[1]
print("M:", M)

n_geogs = geog_data.shape[0] 
n_individuals = geog_data.population.values


# random order of individuals
geog_data.population.sum() # 39M


# for testing
import importlib
importlib.reload(sys.modules['vax_entities'])
importlib.reload(sys.modules['assignment_funcs'])
from vax_entities import Individual, Geog, Location
from assignment_funcs import initialize, compute_ranking, random_fcfs, sequential, reset_assignments, assignment_stats


geogs, locations = initialize(distmatrix=distmatrix, locmatrix=locmatrix, distcoef=distcoef, abd=abd, capacity=10000, M=M, n_individuals=n_individuals)

indiv_ordering = [(tt,ii) for tt in range(len(geogs)) for ii in range(n_individuals[tt])]
np.random.shuffle(indiv_ordering)

random_fcfs(geogs, locations, indiv_ordering)

#===================================================================================================
# Make inputs for demand estimation
#===================================================================================================

# df
# agent_data



geogs[100000].individuals[0]

geogs[100000].individuals[0].geog_id
geogs[100000].individuals[0].epsilon_ij
geogs[100000].individuals[0].u_ij
geogs[100000].individuals[0].location_ranking
geogs[100000].individuals[0].locations_ranked
geogs[100000].individuals[0].location
geogs[100000].individuals[0].rank_assigned


