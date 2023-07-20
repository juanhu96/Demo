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
outpath = f"{datadir}/Intermediate/ca_blk_pharm_dist.csv"

output = subprocess.run(["stata-mp", "-b", "do", f"{current_directory}/Demand/datawork/geonear_pharmacies.do", baselocpath, pharmlocpath, outpath], capture_output=True, text=True)
print(output.stdout)
print(output.stderr)


# check output
blk_dist = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv")
print(blk_dist.km_to_nid.describe())

