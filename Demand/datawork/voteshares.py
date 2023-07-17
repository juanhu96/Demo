# Intersect 2020 precinct shapefile with ZCTA (ZIP Code) shapefile to get vote shares by ZIP 

import geopandas as gpd
datadir = "/export/storage_covidvaccine/Data"

# read in CA precinct shapefile
precincts = gpd.read_file(f"{datadir}/Raw/ca_vest_16/ca_vest_16.shp")
precincts
# tally votes
votecols = [col for col in precincts.columns if 'G16PRE' in col]
precincts['sum_votes'] = precincts[votecols].sum(axis=1)
precincts = precincts.loc[precincts['sum_votes']>0, :]
precincts = precincts.rename(columns={'SRPREC': 'precinct', 'G16PREDCli': 'dvotes', 'G16PRERTru': 'rvotes'})
precincts = precincts[['precinct', 'dvotes', 'rvotes', 'sum_votes', 'geometry']]

#### TRACT ####
# read in tract shapefile
tracts = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2010_06_tract10/tl_2010_06_tract10.shp")
tracts = tracts[['GEOID10', 'geometry']]
tracts = tracts.rename(columns={'GEOID10': 'tract'})
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
print(tractagg.describe())

# write to file
tractagg.to_csv(f"{datadir}/Intermediate/tract_votes.csv", index=False)
print("Done with tract votes")

#### ZIP ####
# read in full national zips
zips = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")
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

print(zipagg.describe())

# write to file
zipagg.to_csv(f"{datadir}/Intermediate/zip_votes.csv", index=False)
print("Done with zip votes")