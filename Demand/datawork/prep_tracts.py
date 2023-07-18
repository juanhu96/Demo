# prepare and merge tract data for demand estimation

import pandas as pd
import numpy as np

datadir = "/export/storage_covidvaccine/Data"


# tract-pharmacy distance pairs
tract_nearest_df = pd.read_csv(f"{datadir}/Intermediate/tract_nearest_dist.csv", dtype={'tract': str}) #from read_tract_dist.py

# tract demographics
tracts = pd.read_csv(f"{datadir}/Intermediate/TRACT_merged.csv", dtype={'tract': str}) #from export_tract.R

print("Columns from TRACT_merged:\n", tracts.columns.tolist())

tracts.rename(columns={'population': 'tr_pop'}, inplace=True)

tracts.isna().sum() # 0 for all columns except hpi and health variables

tracts.loc[tracts['tr_pop'] == 0, 'tr_pop'] = 1 # for computing weights later

# impute vote shares
tracts['dshare'] = tracts['dshare'].fillna(tracts['dshare'].mean())

# impute health variables TODO: check
for vv in ['health_employer','health_medicare','health_medicaid','health_other']: 
    tracts[vv] = tracts[vv].fillna(tracts[vv].mean())

# impute HPI
# 
# 'drop' or 'bottom' or 'nearest' or  TODO: switch
impute_hpi_method = 'drop' 
# not using 2011 since it doesn't add that many observations
if impute_hpi_method == 'drop':
    tracts = tracts.loc[tracts['hpi'].notnull(), :]
elif impute_hpi_method == 'bottom': #assign bottom quantile
    tracts.loc[tracts['hpi'].isnull(), 'hpi'] = 0
elif impute_hpi_method == 'nearest': #use KDTree to find nearest neighbor
    from scipy.spatial import KDTree
    known_hpi_coords = tracts.loc[tracts['hpi'].notna(), ['latitude', 'longitude']]
    unknown_hpi_coords = tracts.loc[tracts['hpi'].isna(), ['latitude', 'longitude']]
    tree = KDTree(known_hpi_coords)
    # find the nearest neighbors for the tracts with missing hpi values
    distances, indices = tree.query(unknown_hpi_coords)
    nearest_hpi_values = tracts.loc[tracts['hpi'].notna(), 'hpi'].iloc[indices].values
    tracts.loc[tracts['hpi'].isna(), 'hpi_filled'] = nearest_hpi_values
    # #inspect
    # tracts.loc[(np.isclose(tracts['hpi'],0.586650)) | (np.isclose(tracts['hpi_filled'],0.586650)), ['hpi', 'hpi_filled', 'latitude', 'longitude']]
    tracts.fillna({'hpi': tracts['hpi_filled']}, inplace=True)
    tracts.drop(columns=['hpi_filled'], inplace=True)



# merge to tract_nearest_df
tracts = tract_nearest_df.merge(tracts, on='tract', how='outer', indicator=True)
print("Merge between distance and demographics:\n", tracts._merge.value_counts()) #perfect match
tracts = tracts.loc[tracts._merge == 'both', :]
tracts.drop(columns=['_merge'], inplace=True)

# merge with tract-ZIP crosswalk - now at the tract-ZIP intersection level
tractzip_cw = pd.read_csv(f"{datadir}/Intermediate/tract_zip_crosswalk.csv", dtype={'zip': str, 'tract': str})

agent_data = tracts.merge(tractzip_cw, on='tract', how='outer', indicator=True)
agent_data['_merge'].value_counts() # 5 left, 20k both
agent_data = agent_data.loc[agent_data._merge == 'both', :]
agent_data.drop(columns=['_merge'], inplace=True)

list(agent_data.columns)


# ZIP-level from prep_zip.py - should not be modified in this script
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv", dtype={'zip': str})

# keep only tracts whose ZIPs are in df
agent_data['zip'].isin(df['zip']).value_counts()
agent_data = agent_data[agent_data['zip'].isin(df['zip'])]

# weights to be the population of the tract over the population in the ZIP
agent_data = agent_data.merge(df[['zip', 'population']].rename(columns={'population':'zip_pop'}), on='zip', how='left')

pop_method = 'tract' # 'tract' or 'zip' TODO: switch
if pop_method == 'tract': # compute weights based on tract population and fraction of the tract's area that is in the ZIP-tract cell
    agent_data['cell_pop'] = agent_data['tr_pop'] * agent_data['frac_of_tract_area']
    agent_data['market_pop'] = agent_data.groupby('zip')['cell_pop'].transform('sum')
    agent_data['weights'] = agent_data['cell_pop'] / agent_data['market_pop']
elif pop_method == 'zip': # compute weights based on ZIP population and fraction of the ZIP's area that is in the ZIP-tract cell
    agent_data['cell_pop'] = agent_data['zip_pop'] * agent_data['frac_of_zip_area']
    agent_data['market_pop'] = agent_data['zip_pop']
    agent_data['weights'] = agent_data['cell_pop'] / agent_data['market_pop']
    # there are NAs because some ZIPs have zero population 


pd.set_option('display.max_columns', None)
agent_data.describe()

# number of tracts for each zip


# there are a few ZIPs in df but not in agent_data - make tracts for these ZIPs that are the ZIPs themselves
print("ZIPs in df but not in agent_data:")
print(df['zip'].isin(agent_data['zip']).value_counts())
df.loc[~df['zip'].isin(agent_data['zip']),'zip'].tolist()



aux_tracts = df.loc[~df['zip'].isin(agent_data['zip']), :].copy()

# make the variables that are needed in the agent_data
aux_tracts = aux_tracts.assign(
    tract = aux_tracts['zip'],
    frac_of_tract_area = 1,
    frac_of_zip_area = 1,
    zip_pop = aux_tracts['population'],
    tr_pop = aux_tracts['population'],
    market_pop = aux_tracts['population'],
    weights = 1,
    nodes = 1
)
aux_tracts['cell_pop'] = [aa if aa > 0 else 1  for aa in aux_tracts['population']]
# get latitude and longitude from zip_coords
zip_coords = pd.read_csv(f"{datadir}/Intermediate/zip_coords.csv", dtype={'zip': str})
aux_tracts= aux_tracts.merge(zip_coords, on='zip', how='left')

aux_tracts = aux_tracts.drop(columns=['vaxfull', 'shares', 'population', 'firm_ids', 'prices'])
set(aux_tracts.columns.tolist()) - set(agent_data.columns.tolist())
set(agent_data.columns.tolist()) - set(aux_tracts.columns.tolist())
# distance - compute euclidean distance between lat/long of ZIP and pharmacy locations
pharmacy_locations = pd.read_csv(f"{datadir}/Raw/Location/00_Pharmacies.csv", usecols=['latitude', 'longitude', 'StateID'])
# subset to CA
pharmacy_locations = pharmacy_locations.loc[pharmacy_locations['StateID'] == 6, :]
pharmacy_locations.drop(columns=['StateID'], inplace=True)
# compute distance 
from haversine import haversine, Unit
def nearest_dist_tract(tt):
    dist_allpharms = [haversine((tt['latitude'], tt['longitude']), (pp['latitude'], pp['longitude']), unit=Unit.KILOMETERS) for pp in pharmacy_locations.to_dict('records')]
    return min(dist_allpharms)

aux_tracts['dist'] = aux_tracts.apply(nearest_dist_tract, axis=1)
# verified that it matches MAR01.csv

# append to agent_data
agent_data = pd.concat([agent_data, aux_tracts], axis=0)

agent_data['market_ids'] = agent_data['zip']
agent_data['nodes'] = 0 # for pyblp (no random coefficients)
agent_data['logdist'] = np.log(agent_data['dist']) 

print("agent_data.describe() \n", agent_data.describe())
print("product data describe() \n", df.describe())


# save to csv
agent_data.to_csv(f"{datadir}/Analysis/Demand/agent_data.csv", index=False)

agent_data.to_csv(f"{datadir}/Analysis/Demand/agent_data_{impute_hpi_method}.csv", index=False)

# summarize tract distance
print("Distance (tract-level):")
print(agent_data['dist'].describe())
print("Log distance (tract-level):")
print(agent_data['logdist'].describe())
print("Distance (mean within ZIP):")
print(agent_data.groupby('market_ids')['dist'].mean().describe())
print("Log distance (mean within ZIP):")
print(agent_data.groupby('market_ids')['logdist'].mean().describe())

# agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")

