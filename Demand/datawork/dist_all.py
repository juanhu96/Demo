# distance between each block and ALL pharmacies
import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Data"
codedir = "/mnt/staff/zhli/VaxDemandDistance"


blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['lat', 'long', 'blkid'])

pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
# subset to CA
pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
pharmacy_locations.drop(columns=['StateID'], inplace=True)

# Convert to NumPy arrays
blocks = blk_coords[['lat', 'long']].values
pharmacies = pharmacy_locations[['latitude', 'longitude']].values

# # subset the blocks and pharmacies
# blocks = blocks[:int(len(blocks)*0.2), :]
# pharmacies = pharmacies[:int(len(pharmacies)*0.2), :]

# Calculate pairwise distances using broadcasting
distances = np.sqrt(((blocks[:, np.newaxis, :] - pharmacies[np.newaxis, :, :]) ** 2).sum(axis=2))

distances.shape

distances_km = distances * 99.28 
# 1 degree of latitude is approximately 111 kilometers.
# 1 degree of longitude is approximately 87.56 km in CA (average latitude of 38°).
# The average of these two values is 99.28 km.

# Save the distances to a npy file
np.save(f"{datadir}/Intermediate/block_pharmacy_distances_all.npy", distances_km)


# verify it's not too far from distdf
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})

import random
id_pick_blk = random.randint(0, blk_coords.shape[0])
id_pick_pharm = random.randint(0,300)
blkid_ = blk_coords.blkid[id_pick_blk] 
pharmid_ = distdf.loc[(distdf['blkid'] == blkid_)].locid.iloc[id_pick_pharm]
distances_km[id_pick_blk, pharmid_]
distdf.dist[(distdf['blkid'] == blkid_) & (distdf['locid'] == pharmid_)]