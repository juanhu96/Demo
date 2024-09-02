#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sep, 2024 by Jingyuan Hu
"""

# distance between each block and ALL current locations
import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Demo/Data"
codedir = "/mnt/phd/jihu/Demo"

blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['lat', 'long', 'blkid'])

locations = pd.read_csv(f"{datadir}/../locations.csv", usecols=['latitude', 'longitude', 'open'])
current_locations = locations.loc[locations['open'] == 1, :]

# Convert to NumPy arrays
blocks = blk_coords[['lat', 'long']].values
current = current_locations[['latitude', 'longitude']].values

# Calculate pairwise distances using broadcasting
distances = np.sqrt(((blocks[:, np.newaxis, :] - current[np.newaxis, :, :]) ** 2).sum(axis=2))
distances.shape
distances_km = distances * 99.28 
# 1 degree of latitude is approximately 111 kilometers.
# 1 degree of longitude is approximately 87.56 km in CA (average latitude of 38°).
# The average of these two values is 99.28 km.

# Save the distances to a npy file
np.save(f"{datadir}/Intermediate/block_current_distances_all.npy", distances_km)


# verify it's not too far from distdf
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_current_dist.csv", dtype={'locid': int, 'blkid': int})
import random
id_pick_blk = random.randint(0, blk_coords.shape[0])
id_pick_current = random.randint(0, 5)
blkid_ = blk_coords.blkid[id_pick_blk] 
currentid_ = distdf.loc[(distdf['blkid'] == blkid_)].locid.iloc[id_pick_current]
print(distances_km[id_pick_blk, currentid_])
print(distdf.dist[(distdf['blkid'] == blkid_) & (distdf['locid'] == currentid_)])