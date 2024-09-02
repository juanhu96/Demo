#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sep, 2024 by Jingyuan Hu
"""

# make crosswalks between blocks and ZIPs and between blocks and tracts
# run after read_block.py

import numpy as np
import pandas as pd
import geopandas as gpd

datadir = "/export/storage_covidvaccine/Demo/Data"

blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['blkid', 'lat', 'long'])

# make a geodataframe
blk_coords = gpd.GeoDataFrame(blk_coords, geometry=gpd.points_from_xy(blk_coords.long, blk_coords.lat))


#### ZIP crosswalk ####
# read in ZIP shapefile
zips = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")[['ZCTA5CE20', 'geometry']]
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})

# find the ZIP containing each block
blk_coords.set_crs(zips.crs, inplace=True)
blk_coords.crs
blk_zip = gpd.sjoin(blk_coords, zips, how="inner", predicate='within')

blk_zip = blk_zip[['lat', 'long', 'zip', 'blkid']]
blk_zip.to_csv(f"{datadir}/Intermediate/blk_zip.csv", index=False)


#### Tract crosswalk ####
# read in tract shapefile
tracts = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2010_06_tract10/tl_2010_06_tract10.shp")
tracts = tracts[['GEOID10', 'geometry']]
tracts['tract'] = tracts['GEOID10'].astype(str).str.slice(start=1) # drop the first digit (0)
tracts.drop(columns=['GEOID10'], inplace=True)

# find the tract containing each block
blk_coords.set_crs(tracts.crs, inplace=True)
blk_tract = gpd.sjoin(blk_coords, tracts, how="inner", predicate='within')

blk_tract = blk_tract[['lat', 'long', 'tract', 'blkid']]
blk_tract.to_csv(f"{datadir}/Intermediate/blk_tract.csv", index=False)


#### Crosswalk between blocks and ZIP-tract intersections ####
# read in ZIP-tract crosswalk (from ziptract.py)
tractzip = gpd.read_file(f"{datadir}/Intermediate/tract_zip_crosswalk.shp")
tractzip = tractzip[['zip', 'tract', 'geometry']]
# find the ZIP-tract intersection containing each block
blk_coords.set_crs(tractzip.crs, inplace=True)
blk_ziptract = gpd.sjoin(blk_coords, tractzip, how="inner", predicate='within')
blk_ziptract = blk_ziptract[['lat', 'long', 'zip', 'tract', 'blkid']]

# merge in population
blk_pop = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['blkid', 'population'])
blk_ziptract = blk_ziptract.drop(columns=['lat', 'long']).merge(blk_pop, on='blkid', how='left')
blk_ziptract = blk_ziptract.rename(columns={'population': 'blk_pop'})

# save
blk_ziptract.to_csv(f"{datadir}/Intermediate/blk_ziptract.csv", index=False)

# ########################
# # read everything
# blk_tract = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv")
# ca_blk_pharm_dist = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv")
# block_data = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
# agent_results = pd.read_csv("/export/storage_covidvaccine/Result/Demand/agent_results_10000_200_3q.csv")

# # merge blk_tract and ca_blk_pharm_dist
# blk_tract.merge(ca_blk_pharm_dist, on='blkid', how='outer', indicator=True)['_merge'].value_counts() # all 

# # merge blk_tract and block_data
# blk_tract.merge(block_data, on='blkid', how='outer', indicator=True)['_merge'].value_counts() # 206 left only

# # merge block_data and agent_results
# block_data.merge(agent_results, on='blkid', how='outer', indicator=True)['_merge'].value_counts() #362 left only

# # merge blk_tract and agent_results
# blk_tract.merge(agent_results, on='blkid', how='outer', indicator=True)['_merge'].value_counts() # 568 left only

# # merge blk_zip and df
# df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
# blk_zip = pd.read_csv(f"{datadir}/Intermediate/blk_zip.csv")
# blk_zip_unique = blk_zip[['zip']].drop_duplicates()
# blk_zip_unique.merge(df, on='zip', how='outer', indicator=True)['_merge'].value_counts() # 49 left only
