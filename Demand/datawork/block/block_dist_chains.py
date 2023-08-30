# find distance to nearest chain store (e.g., Dollar stores) for each block
#
import pandas as pd
import numpy as np
import geopandas as gpd
import subprocess
datadir = "/export/storage_covidvaccine/Data"
# codedir = "/mnt/staff/zhli/VaxDemandDistance" #TODO: changex
codedir = "/mnt/phd/jihu/VaxDemandDistance"

# read chain locations, subset to CA
chain_type = '01_DollarStores'
chain_name = 'Dollar'
chain_locations = pd.read_csv(f"{datadir}/Raw/Location/{chain_type}.csv", usecols=['Latitude', 'Longitude', 'State'])
chain_locations.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
chain_locations = chain_locations.loc[chain_locations['State'] == 'CA', :]
chain_locations.drop(columns=['State'], inplace=True)

chain_locations['id'] = range(chain_locations.shape[0])
chainlocpath = f"{datadir}/Intermediate/ca_{chain_name}_locations.dta"
chain_locations.to_stata(chainlocpath, write_index=False)

# call stata program to compute distance
baselocpath = f"{datadir}/Intermediate/blk_coords.dta"
chainlocpath = f"{datadir}/Intermediate/ca_{chain_name}_locations.dta"

# inspect chain locations
chain = pd.read_stata(chainlocpath)
chain.tail()
len(set(chain.id))

print("Entering Stata...")

# chain distances
outpath = f"{datadir}/Intermediate/ca_blk_{chain_name}_dist.csv"
within = 500 # km
limit = 50 # number of chain stores to consider
output = subprocess.run(["stata-mp", "-b", "do", f"{codedir}/Demand/datawork/geonear_pharmacies.do", baselocpath, chainlocpath, outpath, str(within), str(limit)], capture_output=True, text=True)

print(output.stdout)
print(output.stderr)

# # check output
blk_dist = pd.read_csv(outpath)
print(blk_dist.shape)

