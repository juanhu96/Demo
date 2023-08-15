# make crosswalks between blocks and ZIPs and between blocks and tracts
# run after read_block.py
import numpy as np
import pandas as pd
import geopandas as gpd

datadir = "/export/storage_covidvaccine/Data/"

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
