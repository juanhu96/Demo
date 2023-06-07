# Intersect ZCTA and tract shapefiles to get ZCTA for each tract

import pandas as pd
import geopandas as gpd


datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data/tracts"

# read in full national zips
zips = gpd.read_file(f"{datadir}/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")
zips = zips[['ZCTA5CE20', 'geometry']]
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})
zips['zip'] = zips['zip'].astype(str)

# read in tract shapefile
tracts = gpd.read_file(f"{datadir}/tl_2020_06_tract/tl_2020_06_tract.shp")
tracts = tracts[['GEOID', 'geometry']]
tracts = tracts.rename(columns={'GEOID': 'tract'})
tracts['tract'] = tracts['tract'].astype(str)

tracts
zips

# intersect
tractzip = gpd.overlay(tracts, zips, how='intersection')
tractzip
sorted(tractzip.tract)[:10]
sorted(tractzip.tract)[-10:]

# 2019 ACS tract demographics (has a ton of variables)
acs_df = pd.read_csv(f"{datadir}/CA_TRACT_demographics.csv", low_memory=False)
acs_df.rename(columns={'GIDTR': 'tract'}, inplace=True)
demog_cols = [cc for cc in acs_df.columns if 'pct' not in cc and 'avg' not in cc]
# demog_cols 
pop_cols = [cc for cc in demog_cols if 'pop' in cc or 'Pop' in cc]
# pop_cols
acs_df['tract'] = acs_df['tract'].astype(str)
acs_df['tract'].apply(len).value_counts() # all 10-digits that start with 6
tract_demog = acs_df[['tract']]
tract_demog = tract_demog.assign(trpop = acs_df['Tot_Population_ACS_14_18'])
# pad tract to 11 digits
tract_demog['tract'] = tract_demog['tract'].apply(lambda x: x.zfill(11))

# merge
tractzipvars = tractzip.merge(tract_demog, on='tract', how='outer', indicator=True)
tractzipvars['_merge'].value_counts()



