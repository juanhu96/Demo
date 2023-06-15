# comparison between 2020 and 2010 tracts
# since the HPI uses 2010 tracts, we'll use 2010 for all tract data.
# We'll use 2020 for zip data, with the tract-ZIP crosswalk built from shapefiles.

import pandas as pd
import geopandas as gpd

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data/tracts"


# read in tract centroids
tracts2020 = pd.read_csv("https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR06.txt", dtype={'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str})
tracts2010 = pd.read_csv("https://www2.census.gov/geo/docs/reference/cenpop2010/tract/CenPop2010_Mean_TR06.txt", dtype={'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str})

def make_tractid(dfrow):
    return f"{dfrow['STATEFP']}{dfrow['COUNTYFP']}{dfrow['TRACTCE']}"

tracts2020['tract'] = tracts2020.apply(make_tractid, axis=1)
tracts2010['tract'] = tracts2010.apply(make_tractid, axis=1)
tracts2020.tract.apply(len).value_counts()
tracts2010.tract.apply(len).value_counts()

merged2020 = tracts2020.merge(hpi, on='tract', how='outer', indicator=True)
merged2020._merge.value_counts()

merged2010 = tracts2010.merge(hpi, on='tract', how='outer', indicator=True)
merged2010._merge.value_counts()

merged2010[merged2010._merge=='right_only']

tracts2020.shape
tracts2010.shape

crosswalk = pd.read_csv("https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/tab20_tract20_tract10_natl.txt", sep='|')
crosswalk.head()

len(set(crosswalk.GEOID_TRACT_10))
len(set(crosswalk.GEOID_TRACT_20))
crosswalk.shape


# read hpi2score.csv
hpi = pd.read_csv(f"{datadir}/hpi2score.csv", dtype={'geoid': str})
hpi.rename(columns={'geoid': 'tract'}, inplace=True)
hpi.tract.apply(len).value_counts()
hpi.tract.head()
