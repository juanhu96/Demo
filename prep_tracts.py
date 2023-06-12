# prepare and merge tract data for demest_tracts.py and demest_rc.py
# Combine the following data sources:
# 1. tract demographics
# 2. tract-HPI
# 3. tract-ZIP crosswalk
# 4. tract-pharmacy distance pairs


import pandas as pd
import numpy as np

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data

# Original ZIP-level data with vaccination rates (product_data)
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv", dtype={'zip': str})
df = df.assign(market_ids = df['zip'], prices = 0)
zip_votes = pd.read_csv(f"{datadir}/tracts/zip_votes.csv", dtype={'zip': str})
list(zip_votes.columns)
zip_votes['dshare'] = zip_votes['dvotes'] / (zip_votes['dvotes'] + zip_votes['rvotes']) #make 2-party vote share
zip_votes = zip_votes[['zip', 'dshare']]
df = df.merge(zip_votes, on='zip', how='outer', indicator=True)
df._merge.value_counts() 
df = df.loc[df._merge == 'both', :]
df.drop(columns=['_merge'], inplace=True)

# Tract (geoid) level data with HPI and HPIQuartile
tract_hpi_df = pd.read_csv(f"{datadir}/tracts/HPItract.csv", dtype={'geoid': str})
tract_hpi_df
tract_hpi_df.rename(columns={'geoid': 'tract'}, inplace=True)
tract_hpi_df['tract'].apply(len).value_counts() # all 10-digits that start with 6
sorted(tract_hpi_df['tract'])


# Tract-ZIP crosswalk
tractzip_cw = pd.read_csv(f"{datadir}/tracts/tract_zip_032022.csv", usecols=['TRACT', 'ZIP'], dtype={'TRACT': str, 'ZIP': str})
# verify that TRACT is 11 digits
tractzip_cw['TRACT'].apply(len).value_counts() # all 11
sorted(tractzip_cw.TRACT)

tractzip_cw = tractzip_cw.assign(statefips = tractzip_cw['TRACT'].str[:2])
# keep CA only
tractzip_cw = tractzip_cw[tractzip_cw['statefips'] == '06']
tractzip_cw = tractzip_cw.assign(countyfips = tractzip_cw['TRACT'].str[2:5])
tractzip_cw = tractzip_cw.assign(tractid = tractzip_cw['TRACT'].str[5:])
# tractzip_cw['tractid']
# Make the Tract column the same as the one in tract_nearest_df (10 digits, start with 6)
tractzip_cw = tractzip_cw.assign(tract = '6' + tractzip_cw['countyfips'] + tractzip_cw['tractid'])
# tractzip_cw = tractzip_cw[['tract', 'ZIP']]
tractzip_cw.sort_values(by='tract', inplace=True)
tractzip_cw['tract'].apply(len).value_counts() # all 10-digits that start with 6
tractzip_cw

# 2019 ACS tract demographics (has a ton of variables)
acs_df = pd.read_csv(f"{datadir}/tracts/CA_TRACT_demographics.csv", low_memory=False)
acs_df.rename(columns={'GIDTR': 'tract'}, inplace=True)
# drop one duplicate tract
acs_df = acs_df.drop_duplicates(subset=['tract'])
demog_cols = [cc for cc in acs_df.columns if 'pct' not in cc and 'avg' not in cc]
# demog_cols 
pop_cols = [cc for cc in demog_cols if 'pop' in cc or 'Pop' in cc]
# pop_cols
acs_df['tract'] = acs_df['tract'].astype(str)
acs_df['tract'].apply(len).value_counts() # all 10-digits that start with 6
tract_demog = acs_df[['tract']]
tract_demog = tract_demog.assign(trpop = acs_df['Tot_Population_ACS_14_18'])

# read tract-pharmacy distance pairs
# pre-processed to the 10 nearest pharmacies for each tract.
pairs_df = pd.read_csv(f"{datadir}/tracts/pairs_filtered.csv", usecols=['Tract', 'Distance'],dtype={'Tract': str, 'Distance': float})
pairs_df
pairs_df.rename(columns={'Tract': 'tract', 'Distance': 'dist'}, inplace=True)
# just the nearest pharmacy for each tract
tract_nearest_df = pairs_df.groupby('tract').head(1)

tract_nearest_df['tract'].apply(len).value_counts() # between 8 and 11
# The tract column is messed up. I think there should be FIPS as the first 5, with only the first digit being the state (6XXXX). Followed by a 5 digit tract ID. 
# TODO: verify if my fix is correct
# look at some examples 
tract_nearest_df.groupby(tract_nearest_df['tract'].apply(len)).apply(lambda x: x.sample(10))
# check that the first digit is always 0
tract_nearest_df['tract'].apply(lambda x: x[0]).value_counts()
tract_nearest_df = tract_nearest_df.assign(countyfips = tract_nearest_df['tract'].str[1:5])
tract_nearest_df['countyfips']
tract_nearest_df['tractid'] = tract_nearest_df['tract'].str[5:]
tract_nearest_df['tractid']
# pad the tractid with 0s
tract_nearest_df['tractid'] = tract_nearest_df['tractid'].apply(lambda x: x.zfill(6))
# combine the countyfips and tractid
tract_nearest_df['tract'] = tract_nearest_df['countyfips'] + tract_nearest_df['tractid']


# TEST tract_nearest_df and tract_zip_cw merge
testmerge = tract_nearest_df.merge(tractzip_cw, on='tract', how='outer', indicator=True)
testmerge._merge.value_counts() 
tract_nearest_df.tract
tractzip_cw.tract

###

# merge tract level data
tract_df = tract_nearest_df.merge(tract_hpi_df, on='tract', how='outer', indicator=True)
tract_df._merge.value_counts() #1.5k left, 1.1k right, 6.7k both
tract_df = tract_df.loc[tract_df._merge == 'both', :]
tract_df.drop(columns=['_merge'], inplace=True)
# merge with tract demographics 
tract_df = tract_df.merge(tract_demog, on='tract', how='outer', indicator=True)
tract_df._merge.value_counts() # 1k right, 7k both
tract_df = tract_df.loc[tract_df._merge == 'both', :]
tract_df.drop(columns=['_merge'], inplace=True)

# merge tract_df with tractzip_cw
agent_data = tractzip_cw.merge(tract_df, on='tract', how='outer', indicator=True)
agent_data._merge.value_counts() # 3k left, 12k both
agent_data = agent_data.loc[agent_data._merge == 'both', :]
agent_data.drop(columns=['_merge'], inplace=True)
agent_data = agent_data.rename(columns={'ZIP': 'zip', 'HPI': 'hpi', 'HPIQuartile': 'hpiquartile'})
# 1158 zips in df aren't in agent_data
df['zip'].isin(agent_data['zip']).value_counts()

###### 
# get the agent_data into pyblp format

list(df.columns)
list(agent_data.columns)
agent_data.describe()
agent_data[agent_data['trpop'].isna()]
tract_demog[tract_demog['tract'].isin(agent_data[agent_data['trpop'].isna()]['tract'])]
len(acs_df.tract)
len(np.unique(acs_df.tract))

# If a ZIP has no tracts, create a fake tract that's just the ZIP's HPI and population
aux_tracts = df[['hpi', 'hpiquartile', 'dist', 'market_ids']][~df['zip'].isin(agent_data['zip'])]
aux_tracts = aux_tracts.assign(weights = 1)
aux_tracts

# weights to be the population of the tract over the sum of the population of all tracts in the ZIP
zip_pop = agent_data.groupby('zip')['trpop'].transform('sum')

agent_data = agent_data.assign(market_ids = agent_data['zip'],
                               weights = agent_data['trpop']/zip_pop)

agent_data = agent_data[['market_ids', 'weights', 'hpi', 'hpiquartile', 'dist', 'zip']]
agent_data = pd.concat([agent_data, aux_tracts], ignore_index=True)

# keep ZIPs that are in df
agent_data = agent_data[agent_data['market_ids'].isin(df['market_ids'])]
agent_data['nodes'] = 0 

# agent_data = agent_data.rename(columns={'dist': 'dist0'})
# save to csv
agent_data.to_csv(f"{datadir}/Analysis/agent_data.csv", index=False)
df.to_csv(f"{datadir}/Analysis/product_data_tracts.csv", index=False)

# number of tracts per ZIP
agent_data.groupby('market_ids').nunique().describe()

# summarize tract distance
agent_data.groupby('market_ids')['dist'].mean().describe()
