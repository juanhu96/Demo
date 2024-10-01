#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sep, 2024 by Jingyuan Hu
"""

# find distance to nearest current locations for each block
# run after read_block.py
import pandas as pd
import numpy as np
import geopandas as gpd
import subprocess
datadir = "/export/storage_covidvaccine/Demo/Data"
codedir = "/mnt/phd/jihu/Demo"

# read current locations
locations = pd.read_csv(f"{datadir}/../locations.csv", usecols=['latitude', 'longitude', 'open'])
current_locations = locations.loc[locations['open'] == 1, :]
current_locations['id'] = range(current_locations.shape[0])
currentlocpath = f"{datadir}/Intermediate/current_locations.dta"
current_locations.to_stata(currentlocpath, write_index=False)

# save block coordinates as stata file
blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['lat', 'long', 'blkid'])
blk_coords.rename(columns={'lat': 'latitude', 'long': 'longitude'}, inplace=True)
baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
blk_coords.to_stata(baselocpath, write_index=False)

# call stata program to compute distance
currentlocpath = f"{datadir}/Intermediate/current_locations.dta"

# inspect pharmacy locations
current = pd.read_stata(currentlocpath)
current.tail()
len(set(current.id))

print("Entering Stata...")
# distances between block and current locations
outpath = f"{datadir}/Intermediate/ca_blk_current_dist.csv"
within = 100 # km
limit = 1000 # number of locations to consider
output = subprocess.run(["stata-mp", "-b", "do", f"{codedir}/Demand/datawork/geonear_current.do", baselocpath, currentlocpath, outpath, str(within), str(limit)], capture_output=True, text=True)

print(output.stdout)
print(output.stderr)

# check output
blk_dist = pd.read_csv(outpath)
print(blk_dist.shape)

# save a version with just the nearest current for demand estimation
blk_dist_nearest = blk_dist[['blkid', 'logdist']].groupby('blkid').min().reset_index()
blk_dist_nearest.to_csv(f"{datadir}/Intermediate/ca_blk_current_dist_nearest.csv", index=False)

# =============================================================================

# find distance to nearest new locations for each block
new_locations = locations.loc[locations['open'] == 0, :]
new_locations['id'] = range(new_locations.shape[0])
newlocpath = f"{datadir}/Intermediate/new_locations.dta"
new_locations.to_stata(newlocpath, write_index=False)

newlocpath = f"{datadir}/Intermediate/new_locations.dta"

new = pd.read_stata(newlocpath)
new.tail()
len(set(new.id))

print("Entering Stata...")

outpath = f"{datadir}/Intermediate/ca_blk_new_dist.csv"
within = 500
limit = 50
output = subprocess.run(["stata-mp", "-b", "do", f"{codedir}/Demand/datawork/geonear_current.do", baselocpath, newlocpath, outpath, str(within), str(limit)], capture_output=True, text=True)

print(output.stdout)
print(output.stderr)

# check output
blk_dist = pd.read_csv(outpath)
print(blk_dist.shape)