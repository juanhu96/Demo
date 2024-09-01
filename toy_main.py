#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug, 2024
@author: Jingyuan Hu
"""

import pandas as pd

#########

datadir = "/export/storage_covidvaccine/Demo/"
area_file_path = f'{datadir}area_file.csv' 
location_file_path = f'{datadir}location_file.csv'
Areas = pd.read_csv(area_file_path, usecols=['ZIP', 'Rates', 'X', 'RC']) # X: unemployment rate, RC: HPI quartile
Locations = pd.read_csv(location_file_path, usecols=['Lat', 'Long', 'Open'])

print(Areas.head(), Locations.head())
# M, K

#########

### NOTES: I WOULD FOCUS ON CALIFORNIA ONLY FOR NOW
### OTHERWISE THIS REQUIRES KEEPING THE SHAPEFILES/COORDINATES/OTHER INFO FOR THE ENTIRE STATE


# GET ALL THE BLOCKS WITHIN THE ZIP


# 1. read_block.py: read in block coordinates and population, output blk_coords_pop.csv (TODO: FOCUS ON THE BLOCK WITHIN THE ZIP)
# 2. block_cw.py: read blk_coords_pop.csv, output blk_ziptract.csv
# 3. block_dist.py: read blk_coords_pop.csv and locations of stores, output ca_blk_pharm_dist.csv
# 4. prep_block.py: prepare and merge block data for demand estimation, output block_data.csv



#########

# RUN THE DEMAND ESTIMATION




#########


