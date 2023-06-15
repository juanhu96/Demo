# Intersect ZIP and tract shapefiles. ZIPs are 2020, tracts are 2010 (for compatibility with HPI and ACS data).

import pandas as pd
import geopandas as gpd


datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data/tracts"

# read in full national zips
zips = gpd.read_file(f"{datadir}/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")
zips.head()
zips = zips[['ZCTA5CE20', 'geometry']]
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})
zips['zip'] = zips['zip'].astype(str)

# read in tract shapefile (CA, 2010)
tracts = gpd.read_file(f"{datadir}/tl_2010_06_tract10/tl_2010_06_tract10.shp")
tracts = tracts[['GEOID10', 'geometry']]
tracts = tracts.rename(columns={'GEOID10': 'tract'})
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
tractzipvars = tractzipvars.drop(columns=['_merge'])

# distances
pairs_df = pd.read_csv(f"{datadir}/pairs_filtered.csv", usecols=['Tract', 'Distance'],dtype={'Tract': str, 'Distance': float})
pairs_df
pairs_df.rename(columns={'Tract': 'tract', 'Distance': 'dist'}, inplace=True)
# just the nearest pharmacy for each tract
tract_nearest_df = pairs_df.groupby('tract').head(1)

# FIX TRACT ID
tract_nearest_df['tract'].apply(len).value_counts() # between 8 and 11
# The tract column is messed up. I think there should be FIPS as the first 5, with only the first digit being the state (6XXXX). Followed by a 5 digit tract ID. 
# TODO: verify if my fix is correct
tract_nearest_df = tract_nearest_df.assign(countyfips = tract_nearest_df['tract'].str[1:6])
tract_nearest_df['tractid'] = tract_nearest_df['tract'].str[6:]
# pad the tractid with 0s
tract_nearest_df['tractid'] = tract_nearest_df['tractid'].apply(lambda x: x.zfill(5))
# combine the countyfips and tractid
tract_nearest_df['tract'] = tract_nearest_df['countyfips'] + tract_nearest_df['tractid']
tract_nearest_df['tract'] 

# MERGE
tractzipvarsdist = tractzipvars.merge(tract_nearest_df, on='tract', how='outer', indicator=True)