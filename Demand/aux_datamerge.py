import numpy as np
import pandas as pd
try:
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import demest_funcs as de


datadir = "/export/storage_covidvaccine/Data"


## ZIP-level data, produced by /Demand/datawork/zip/prep_zip.py
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
nsplits = 3 # number of HPI groups
df = de.hpi_dist_terms(df, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)

## block-ZIP crosswalk
cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv") #produced by /Demand/datawork/block/prep_block.py
cw_pop.columns.tolist()
cw_pop.sort_values(by=['blkid'], inplace=True)

## block-pharmacy distances (~500 rows per block, >1B rows)
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
#produced by /Demand/datawork/block/block_dist.py, which calls /Demand/datawork/geonear_pharmacies.do
# NOTE: the locid is some auxiliary pharmacy ID produced (as "id") at the start of block_dist.py

## block-level data
agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
agent_data_read.columns.tolist()
# the dist column is the distance to the nearest pharmacy


# add HPI quantile to agent data
hpi_level = 'tract'
if hpi_level == 'zip':
    agent_data_read = agent_data_read.merge(df[['market_ids', 'hpi_quantile']], on='market_ids', how='left')
elif hpi_level == 'tract':
    tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") #from prep_tracts.py
    blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) #from block_cw.py
    splits = np.linspace(0, 1, nsplits+1)
    agent_data_read = agent_data_read.merge(blk_tract_cw, on='blkid', how='left')
    tract_hpi['hpi_quantile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1
    agent_data_read = agent_data_read.merge(tract_hpi[['tract', 'hpi_quantile']], on='tract', how='left')


agent_data_full = distdf.merge(agent_data_read[['blkid', 'market_ids', 'hpi_quantile']], on='blkid', how='left')

agent_data_full = de.hpi_dist_terms(agent_data_full, nsplits=nsplits, add_bins=False, add_dummies=True, add_dist=True)
