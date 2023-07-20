# prepare and merge block data for demand estimation

import pandas as pd
import numpy as np

datadir = "/export/storage_covidvaccine/Data"

distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", usecols=['id', 'km_to_nid'])
distdf.rename(columns={'km_to_nid': 'dist', 'id': 'blkid'}, inplace=True)

cw = pd.read_csv(f"{datadir}/Intermediate/blk_zip.csv", usecols=['zip', 'blkid'])

popdf = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['blkid', 'population'])

# merge distance and ZIP-crosswalk data
blocks = cw.merge(distdf, on = 'blkid', how='outer', indicator=True)
blocks['_merge'].value_counts() # 206 right_only - blocks that are not in a ZIP
blocks = blocks.loc[blocks['_merge'] == 'both']
blocks = blocks.drop(columns=['_merge'])

# merge population data
blocks = blocks.merge(popdf, on = 'blkid', how='outer', indicator=True)
blocks['_merge'].value_counts() # 206 right_only - blocks that are not in a ZIP
blocks = blocks.loc[blocks['_merge'] == 'both']
blocks = blocks.drop(columns=['_merge'])

# weights
blocks['weights'] = blocks['population']/blocks.groupby('zip')['population'].transform('sum')
blocks.sort_values(by=['zip', 'weights'], inplace=True)

# miscellaneous variables
blocks = blocks.assign(
    market_ids = blocks['zip'],
    nodes = 1,
    logdist = np.log(blocks['dist'])
)

# save
blocks.to_csv(f"{datadir}/Analysis/Demand/block_data.csv", index=False)


# agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")
# agent_data.columns
