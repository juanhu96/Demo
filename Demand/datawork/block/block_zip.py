# find a ZIP for each block group
# run after read_block.py
import numpy as np
import pandas as pd
import geopandas as gpd

datadir = "/export/storage_covidvaccine/Data/"

blk_coords = pd.read_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", usecols=['blkid', 'lat', 'long'])

# make a geodataframe
blk_coords = gpd.GeoDataFrame(blk_coords, geometry=gpd.points_from_xy(blk_coords.long, blk_coords.lat))

# read in ZIP shapefile
zips = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")[['ZCTA5CE20', 'geometry']]
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})

# find the ZIP containing each block
blk_coords.set_crs(zips.crs, inplace=True)
blk_coords.crs
blk_zip = gpd.sjoin(blk_coords, zips, how="inner", predicate='within')

blk_zip = blk_zip[['lat', 'long', 'zip', 'blkid']]

# save
blk_zip.to_csv(f"{datadir}/Intermediate/blk_zip.csv", index=False)


