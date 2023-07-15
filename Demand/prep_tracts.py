# prepare and merge tract data for demest_tracts.py and demest_rc.py
# Combine the following data sources:
# 1. tract demographics
# 2. tract-HPI
# 3. tract-ZIP crosswalk (crerated by ziptract.py)
# 4. tract-pharmacy distance pairs

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
impute_hpi_method = 'drop' # 'drop' or 'bottom' or 'nearest' or  TODO: switch
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
df['zip'].isin(agent_data['zip']).value_counts() #8 ZIPs in df but not in agent_data
df.columns.tolist()
# keep tracts that are in df
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

# TODO: there are 8 ZIPs in df but not in agent_data

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

