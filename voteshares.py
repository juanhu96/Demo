# Intersect 2020 precinct shapefile with ZCTA (ZIP Code) shapefile to get vote shares by ZIP 

import geopandas as gpd
datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data/tracts"

# read in CA precinct shapefile
precincts = gpd.read_file(f"{datadir}/ca_vest_20/ca_vest_20.shp")

# tally votes
votecols = [col for col in precincts.columns if 'G20PRE' in col]
precincts['sum_votes'] = precincts[votecols].sum(axis=1)
precincts = precincts.loc[precincts['sum_votes']>0, :]
precincts = precincts.rename(columns={'SRPREC': 'precinct', 'G20PREDBID': 'dvotes', 'G20PRERTRU': 'rvotes'})
precincts = precincts[['precinct', 'dvotes', 'rvotes', 'sum_votes', 'geometry']]


#### 1. ZIP ####
# read in full national zips
zips = gpd.read_file(f"{datadir}/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")
zips = zips[['ZCTA5CE20', 'geometry']]
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})
zips['zip'] = zips['zip'].astype(str)

# overlay precincts and zips
zips = zips.to_crs(precincts.crs)
preczip = gpd.overlay(precincts, zips, how='intersection')
# get area of each precinct to divvy up votes
preczip['area'] = preczip.geometry.area
total_precinct_area = preczip.groupby('precinct')['area'].sum()
preczip['area_fraction'] = preczip['area'] / preczip['precinct'].map(total_precinct_area)

preczip['dvotes'] = preczip['dvotes'] * preczip['area_fraction']
preczip['rvotes'] = preczip['rvotes'] * preczip['area_fraction']
preczip['sum_votes'] = preczip['sum_votes'] * preczip['area_fraction']

# aggregate to zip
zipagg = preczip[['zip', 'dvotes', 'rvotes', 'sum_votes']].groupby('zip').sum().reset_index()
zipagg['dshare'] = zipagg['dvotes'] / zipagg['sum_votes']
zipagg['rshare'] = zipagg['rvotes'] / zipagg['sum_votes']
zipagg.describe()

# write to file
zipagg.to_csv(f"{datadir}/zip_votes.csv", index=False)


#### 2. TRACT ####
# read in tract shapefile
tracts = gpd.read_file(f"{datadir}/tl_2020_06_tract/tl_2020_06_tract.shp")
tracts = tracts[['GEOID', 'geometry']]
tracts = tracts.rename(columns={'GEOID': 'tract'})
tracts['tract'] = tracts['tract'].astype(str)

# overlay precincts and tracts
tracts = tracts.to_crs(precincts.crs)
prectract = gpd.overlay(precincts, tracts, how='intersection')
# get area of each precinct to divvy up votes
prectract['area'] = prectract.geometry.area
total_precinct_area = prectract.groupby('precinct')['area'].sum()
prectract['area_fraction'] = prectract['area'] / prectract['precinct'].map(total_precinct_area)

prectract['dvotes'] = prectract['dvotes'] * prectract['area_fraction']
prectract['rvotes'] = prectract['rvotes'] * prectract['area_fraction']
prectract['sum_votes'] = prectract['sum_votes'] * prectract['area_fraction']

# aggregate to tract
tractagg = prectract[['tract', 'dvotes', 'rvotes', 'sum_votes']].groupby('tract').sum().reset_index()
tractagg['dshare'] = tractagg['dvotes'] / tractagg['sum_votes']
tractagg['rshare'] = tractagg['rvotes'] / tractagg['sum_votes']
tractagg.describe()

# write to file
tractagg.to_csv(f"{datadir}/tract_votes.csv", index=False)