# find distance to nearest pharmacy for each block
#
import pandas as pd
import numpy as np
import geopandas as gpd
import subprocess
import os
datadir = "/export/storage_covidvaccine/Data"


# save block coordinates as stata file
blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['lat', 'long', 'blkid'])
blk_coords.rename(columns={'lat': 'latitude', 'long': 'longitude', 'blkid': 'id'}, inplace=True)
baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
blk_coords.to_stata(baselocpath, write_index=False)


# call stata program to compute distance

current_directory = os.getcwd()
print(current_directory)
pharmlocpath = f"{datadir}/Intermediate/ca_pharmacy_locations.dta"

# nearest N pharmacies
N = 10
outpath = f"{datadir}/Intermediate/ca_blk_pharm_dist_{N}.csv"
output = subprocess.run(["stata-mp", "-b", "do", f"{current_directory}/Demand/datawork/geonear_pharmacies.do", baselocpath, pharmlocpath, outpath, str(N)], capture_output=True, text=True)

# check output
blk_dist = pd.read_csv(outpath)
blk_dist = blk_dist.drop(columns=['latitude', 'longitude']).rename(columns={'id': 'blkid'})
# overwrite the output
blk_dist.to_csv(outpath, index=False)

# save a version with just the nearest pharmacy for demand estimation
blk_dist_nearest = blk_dist[['blkid', 'km_to_nid1']].rename(columns={'km_to_nid1': 'dist'})
f"{datadir}/Intermediate/ca_blk_pharm_dist_nearest.csv"


