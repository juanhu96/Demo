#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug, 2024
@author: Jingyuan Hu

datadir = "/export/storage_covidvaccine/Demo/Data"

#########################################################################################

CONNECT BLOCK, TRACT, ZIP FOR ENTIRE STATE

1. read_block.py: read in block coordinates and population, save as blk_coords_pop.csv

2. ziptract.py: read in ZIP and tract shapefiles, make crosswalk between ZIPs and tracts

3. block_cw.py: read blk_coords_pop.csv, make crosswalks between blocks and ZIPs and between blocks and tracts

#########################################################################################

COMPUTE DISTANCES BETWEEN BLOCKS AND CURRENT LOCATIONS

4. block_dist.py: find distance to nearest locations for each block (requires locations.csv)

5. dist_all.py: distance between each block and ALL current locations

#########################################################################################

PREPARATION FOR DEMAND ESTIMATION

6. prep_block.py: prepare and merge block data for demand estimation

7. prep_zip.py: this subsumes prep_demest.do and trace back to raw data (requires areas.csv)

#########################################################################################

DEMAND ESTIMATION

8. demest_assm.py: demand estimation with capacity constraints
python3 Demand/demest_assm.py 10000 5 4 mnl


"""

import pandas as pd

#########

# datadir = "/export/storage_covidvaccine/Demo/"
# area_file_path = f'{datadir}area_file.csv' 
# location_file_path = f'{datadir}location_file.csv'
# Areas = pd.read_csv(area_file_path, usecols=['ZIP', 'Rates', 'X', 'RC']) # X: unemployment rate, RC: HPI quartile
# Locations = pd.read_csv(location_file_path, usecols=['Lat', 'Long', 'Open'])

# print(Areas.head(), Locations.head())
# M, K

#########

### NOTES: I WOULD FOCUS ON CALIFORNIA ONLY FOR NOW
### OTHERWISE THIS REQUIRES KEEPING THE SHAPEFILES/COORDINATES/OTHER INFO FOR THE ENTIRE STATE


# GET ALL THE BLOCKS WITHIN THE ZIP



#########

# RUN THE DEMAND ESTIMATION




#########


