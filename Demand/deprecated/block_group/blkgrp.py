import pandas as pd
import pygris
from census import Census
from us import states
import geopandas as gpd

datadir = "/export/storage_covidvaccine/Data/"

# read in block group shapefile
bg = pygris.block_groups(state='CA', year=2020)
bg = bg.rename(columns={'GEOID': 'bg', 'INTPTLAT': 'latitude', 'INTPTLON': 'longitude'})
bg = bg[['bg', 'latitude', 'longitude', 'geometry']]


# block group population
c = Census("4a958a9b702db75e0b615c52c8d87287768deb44")
bgpop = c.acs5.get('B01001_001E', geo={'for':'block group:*', 'in':'state:06 county:* tract:*'} )
bgpop = pd.DataFrame(bgpop).rename(columns={'B01001_001E':'pop'})
bgpop.head()
bgpop.shape
# save
bgpop.to_csv(f"{datadir}/Intermediate/bg_pop.csv", index=False)



# intersect with zip shape
zips = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})
zips = zips[['zip', 'geometry']]
zips['zip'] = zips['zip'].astype(str)
zips.head()

bgzip = gpd.overlay(bg, zips, how='intersection')
bgzip.head()
bgzip.shape
bgzip.rename(columns={'geoid': 'bg'}, inplace=True)
# number of block groups per zip
bgzip.groupby('zip').size().describe()

# number of zips per block group
bgzip.groupby('bg').size().describe()

# compute the block group population in the particular ZIP, based on the proportion of the block group area that is in the ZIP

bgzip = bgzip.to_crs(3857) #TODO: check if this is the right CRS
bgzip['area'] = bgzip.geometry.area
for col in ['bg', 'zip']:
    col_area = bgzip.groupby(col)['area'].sum()
    bgzip[f"frac_of_{col}_area"] = bgzip['area'] / bgzip[col].map(col_area)

bgzip_out = bgzip.drop(columns=['geometry']).sort_values(by=['zip', 'bg'])
bgzip_out.head()
bgzip_out.to_csv(f"{datadir}/Intermediate/bg_zip_crosswalk.csv", index=False)
