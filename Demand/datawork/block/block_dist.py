# find distance to nearest pharmacy for each block
#
import pandas as pd
import numpy as np
import geopandas as gpd
import subprocess
datadir = "/export/storage_covidvaccine/Data"
codedir = "/mnt/staff/zhli/VaxDemandDistance" #TODO: change

# read pharmacy locations
pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
# subset to CA
pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
pharmacy_locations.drop(columns=['StateID'], inplace=True)
pharmacy_locations['id'] = range(pharmacy_locations.shape[0])
pharmlocpath = f"{datadir}/Intermediate/ca_pharmacy_locations.dta"
pharmacy_locations.to_stata(pharmlocpath, write_index=False)


# save block coordinates as stata file
blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['lat', 'long', 'blkid'])
blk_coords.rename(columns={'lat': 'latitude', 'long': 'longitude'}, inplace=True)
baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
blk_coords.to_stata(baselocpath, write_index=False)


# call stata program to compute distance

pharmlocpath = f"{datadir}/Intermediate/ca_pharmacy_locations.dta"

# inspect pharmacy locations
pharm = pd.read_stata(pharmlocpath)
pharm.tail()
len(set(pharm.id))

print("Entering Stata...")

# pharmacy distances
outpath = f"{datadir}/Intermediate/ca_blk_pharm_dist.csv"
within = 200 # km
limit = 1000 # number of pharmacies to consider
output = subprocess.run(["stata-mp", "-b", "do", f"{codedir}/Demand/datawork/geonear_pharmacies.do", baselocpath, pharmlocpath, outpath, str(within)], capture_output=True, text=True)

print(output.stdout)
print(output.stderr)

# # check output
blk_dist = pd.read_csv(outpath)
print(blk_dist.shape)


# save a version with just the nearest pharmacy for demand estimation
blk_dist_nearest = blk_dist[['blkid', 'logdist']].groupby('blkid').min().reset_index()
blk_dist_nearest.to_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist_nearest.csv", index=False)

