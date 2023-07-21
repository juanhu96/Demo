# compute distances from each block group to nearest pharmacy

import pandas as pd
import pygris

datadir = "/export/storage_covidvaccine/Data"

# read pharmacy locations
pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
# subset to CA
pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
pharmacy_locations.drop(columns=['StateID'], inplace=True)
pharmacy_locations['id'] = pharmacy_locations.index
pharmlocpath = f"{datadir}/Intermediate/ca_pharmacy_locations.dta"
pharmacy_locations.to_stata(pharmlocpath, write_index=False)

# read in block group coordinates
bg = pygris.block_groups(state='CA', year=2020)
bg = bg.rename(columns={'GEOID': 'id', 'INTPTLAT': 'latitude', 'INTPTLON': 'longitude'})
bg = bg[['id', 'latitude', 'longitude']]
bg.latitude = bg.latitude.astype(float)
bg.longitude = bg.longitude.astype(float)
baselocpath = f"{datadir}/Intermediate/ca_bg_coords.dta"
bg.to_stata(baselocpath, write_index=False)


# call stata program to compute distance

import subprocess
import os
datadir = "/export/storage_covidvaccine/Data"
current_directory = os.getcwd()
print(current_directory)
pharmlocpath = f"{datadir}/Intermediate/ca_pharmacy_locations.dta"
baselocpath = f"{datadir}/Intermediate/ca_bg_coords.dta"

outpath = f"{datadir}/Intermediate/ca_bg_pharm_dist.csv"
output = subprocess.run(["stata-mp", "-b", "do", f"{current_directory}/Demand/datawork/geonear_pharmacies.do", baselocpath, pharmlocpath, outpath], capture_output=True, text=True)
print(output.stdout)
print(output.stderr)

