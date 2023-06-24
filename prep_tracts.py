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
usepairs = False
if usepairs:
    # pre-processed to the 10 nearest pharmacies for each tract.
    pairs_df = pd.read_csv(f"{datadir}/tracts/pairs_filtered.csv", usecols=['Tract', 'Distance'],dtype={'Tract': str, 'Distance': float})
    pairs_df['Distance'].describe()
    pairs_df
    pairs_df.rename(columns={'Tract': 'tract', 'Distance': 'dist'}, inplace=True)
    # just the nearest pharmacy for each tract
    tract_nearest_df = pairs_df.groupby('tract').head(1)
else:
    tract_nearest_df = pd.read_csv(f"{datadir}/Intermediate/tract_nearest_dist.csv", dtype={'tract': str}) #from tractdist.py

tract_nearest_df['tract'].apply(len).value_counts() # 11
tract_nearest_df['tract'] = tract_nearest_df['tract'].str[1:] # drop the first digit


# tract hpi
tract_hpi_df = pd.read_csv(f"{datadir}/Raw/hpi2score.csv", dtype={'geoid': str}, usecols=['geoid', 'value', 'percentile'])
tract_hpi_df.drop(columns=['percentile'], inplace=True)
tract_hpi_df.sort_values(by='value', inplace=True)
tract_hpi_df['hpi_quartile'] = pd.qcut(tract_hpi_df['value'], 4, labels=False) + 1
tract_hpi_df.rename(columns={'geoid': 'tract', 'value': 'hpi'}, inplace=True)
tract_hpi_df.tract.apply(len).value_counts() # all 11 digits
tract_hpi_df['tract'] = tract_hpi_df['tract'].str[1:] # drop the first digit

# merge hpi and nearest pharmacy
tracts = tract_nearest_df.merge(tract_hpi_df, on='tract', how='outer', indicator=True)
tracts._merge.value_counts()
tracts = tracts.loc[tracts._merge != 'right_only', :] #only one right_only
tracts.drop(columns=['_merge'], inplace=True)

# impute HPI for tracts that don't have it
tracts['hpi'] = tracts['hpi'].fillna(tracts['hpi'].mean())
tracts['hpi_quartile'] = tracts['hpi_quartile'].fillna(2) 


# tract demographics
tract_demog = pd.read_csv(f"{datadir}/Raw/notreallyraw/TRACT_merged.csv", dtype={'tract': str})
# testmerge = tract_demog.merge(tract_nearest_df, on='tract', how='outer', indicator=True)
# testmerge._merge.value_counts() # perfect match with centroids
print(tract_demog.columns.tolist())
tract_demog.columns = tract_demog.columns.str.lower()
tract_demog.rename(columns={'population': 'tr_pop'}, inplace=True)
tract_demog.drop(columns=['state_id', 'county_id', 'tract_id', 'hpi', 'hpiquartile', 'dshare', 'rshare', 'dvotes', 'rvotes', 'sum_votes', 'latitude', 'longitude', 'land_area', 'health_none', 'race_white'], inplace=True) #TODO: re-construct these things 
tract_demog['tract'] 
tract_demog['tract'].apply(lambda x: x[0]).value_counts() 
tract_demog['tract'].apply(len).value_counts() # all 10-digits that start with 6
tracts = tracts.merge(tract_demog, on='tract', how='outer', indicator=True)
print(tracts._merge.value_counts()) #perfect match
tracts = tracts.loc[tracts._merge != 'right_only', :]
tracts.drop(columns=['_merge'], inplace=True)
# impute health variables
for vv in ['health_employer','health_medicare','health_medicaid','health_other']:
    tracts[vv] = tracts[vv].fillna(tracts[vv].mean())



# merge with tract-level vote shares
tract_votes = pd.read_csv(f"{datadir}/Intermediate/tract_votes.csv", usecols=['tract', 'dshare'], dtype={'tract': str})
tract_votes['tract'].apply(len).value_counts() # 11
tract_votes['tract'].apply(lambda x: x[0]).value_counts() # all 0
tract_votes['tract'] = tract_votes['tract'].str[1:] # drop the first digit

tracts = tracts.merge(tract_votes, on='tract', how='outer', indicator=True)
tracts._merge.value_counts()
tracts = tracts.loc[tracts._merge != 'right_only', :]
tracts.drop(columns=['_merge'], inplace=True)
# impute vote shares
tracts['dshare'] = tracts['dshare'].fillna(tracts['dshare'].mean())


# merge with tract-ZIP crosswalk
tractzip_cw = pd.read_csv(f"{datadir}/Intermediate/tract_zip_crosswalk.csv", dtype={'zip': str, 'tract': str})
# verify that TRACT is 11 digits
tractzip_cw['tract'].apply(len).value_counts() # all 11
tractzip_cw['tract'] = tractzip_cw['tract'].str[1:] # drop the first digit
tractzip_cw['tract'].apply(len).value_counts() # all 10-digits that start with 6


agent_data = tracts.merge(tractzip_cw, on='tract', how='outer', indicator=True)
agent_data['_merge'].value_counts() # 5 left, 20k both
agent_data = agent_data.loc[agent_data._merge == 'both', :]
agent_data.drop(columns=['_merge'], inplace=True)

list(agent_data.columns)
agent_data.describe()

# Original ZIP-level data with vaccination rates (product_data) - should not be modified in this script
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv", dtype={'zip': str})
df['zip'].isin(agent_data['zip']).value_counts()
df = df.assign(hpi_quartile = df['hpiquartile']) #patch for consistency with other quartile-based variables
# keep ZIPs that are in df
agent_data = agent_data[agent_data['zip'].isin(df['zip'])]

# approximate race and income for ZIPs that double as its own tract
df = df.assign(race = 1-df.race_black-df.race_asian-df.race_hispanic-df.race_other)
df = df.assign(income = df.medianhhincome)


# weights to be the population of the tract over the sum of the population of all tracts in the ZIP
zip_pop = agent_data.groupby('zip')['tr_pop'].transform('sum')
agent_data = agent_data.assign(market_ids = agent_data['zip'],
                               weights = agent_data['tr_pop']/zip_pop)

agent_data.describe()
print(agent_data.columns.tolist())
agent_data[['hpi']].describe()
agent_data_cols = ['market_ids', 'hpi', 'hpi_quartile', 'dist', 'zip', 'race', 'income'] + tract_demog.columns.tolist()


agent_data_cols = [cc.lower() for cc in agent_data_cols]
len(agent_data_cols)
len(set(agent_data_cols))
# agent_data = agent_data[agent_data_cols]

# If a ZIP has no tracts, create a fake tract that's just the ZIP
zips_wotracts = df.loc[~df['zip'].isin(agent_data['zip'])]

aux_tracts = zips_wotracts[['zip', 'dshare', 'dist',
       'race_black', 'race_asian', 'race_hispanic', 
       'race_other', 'health_employer', 'health_medicare', 
       'health_medicaid', 'health_other', 'collegegrad', 
       'medianhhincome', 'poverty', 'unemployment',
       'medianhomevalue', 'hpi', 'popdensity', 'market_ids', 'hpi_quartile'
       ]].assign(tract = zips_wotracts['zip'],
                 weights = 1,
                 tr_pop = zips_wotracts['population'])

pd.set_option('display.max_columns', None)
agent_data = pd.concat([agent_data, aux_tracts], ignore_index=True)

agent_data['nodes'] = 0 # for pyblp (no random coefficients)
agent_data['logdist'] = np.log(agent_data['dist']) 

print("agent_data.describe() \n", agent_data.describe())

# save to csv
agent_data.to_csv(f"{datadir}/Analysis/agent_data.csv", index=False)

# summarize tract distance
print(agent_data.groupby('market_ids')['dist'].mean().describe())
