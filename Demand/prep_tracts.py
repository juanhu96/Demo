# prepare and merge tract data for demest_tracts.py and demest_rc.py
# Combine the following data sources:
# 1. tract demographics
# 2. tract-HPI
# 3. tract-ZIP crosswalk (crerated by ziptract.py)
# 4. tract-pharmacy distance pairs

import pandas as pd
import numpy as np

datadir = "/export/storage_covidvaccine/Data"

useold = False #replicate demandest_0622
if useold:
    useolddemog = True
else:
    useolddemog = False


# tract-pharmacy distance pairs
tract_nearest_df = pd.read_csv(f"{datadir}/Intermediate/tract_nearest_dist.csv", dtype={'tract': str}) #from read_tract_dist.py
tract_nearest_df['tract'].apply(len).value_counts() 


# tract hpi
tract_hpi_df = pd.read_csv(f"{datadir}/Raw/hpi_tract_2022.csv", dtype={'geoid': str}, usecols=['geoid', 'value', 'percentile'])
tract_hpi_df.sort_values(by='value', inplace=True)
tract_hpi_df['hpi_quartile'] = pd.qcut(tract_hpi_df['value'], 4, labels=False) + 1
tract_hpi_df.rename(columns={'geoid': 'tract', 'value': 'hpi'}, inplace=True)
tract_hpi_df.tract.apply(len).value_counts() # all 11 digits
tract_hpi_df['tract'] = tract_hpi_df['tract'].str[1:] # drop the first digit
# merge hpi and nearest pharmacy
tracts = tract_nearest_df.merge(tract_hpi_df, on='tract', how='outer', indicator=True)
print("Distance to HPI merge:\n", tracts._merge.value_counts())
tracts = tracts.loc[tracts._merge != 'right_only', :] #only one right_only
tracts.drop(columns=['_merge'], inplace=True)

# tract demographics
if useolddemog:
    acs_df = pd.read_csv(f"{datadir}/Raw/notreallyraw/CA_TRACT_demographics.csv", low_memory=False)
    acs_df.rename(columns={'GIDTR': 'tract'}, inplace=True)
    # drop one duplicate tract
    acs_df = acs_df.drop_duplicates(subset=['tract'])
    acs_df['tract'] = acs_df['tract'].astype(str)
    acs_df['tract'].apply(len).value_counts() # all 10-digits that start with 6
    tract_demog = acs_df[['tract']]
    tract_demog = tract_demog.assign(tr_pop = acs_df['Tot_Population_ACS_14_18'], race = acs_df['pct_NH_White_alone_ACS_14_18'], income = acs_df['Med_HHD_Inc_ACS_14_18'].replace('[\$,]', '', regex=True).astype(float))
    tract_demog.columns = tract_demog.columns.str.lower()

else:
    tract_demog = pd.read_csv(f"{datadir}/Raw/notreallyraw/TRACT_merged.csv", dtype={'tract': str})
    # testmerge = tract_demog.merge(tract_nearest_df, on='tract', how='outer', indicator=True)
    # testmerge._merge.value_counts() # perfect match with centroids
    tract_demog.columns = tract_demog.columns.str.lower()
    tract_demog.rename(columns={'population': 'tr_pop'}, inplace=True)
    tract_demog.drop(columns=['state_id', 'county_id', 'tract_id', 'hpi', 'hpiquartile', 'dshare', 'rshare', 'dvotes', 'rvotes', 'sum_votes', 'latitude', 'longitude', 'land_area', 'health_none', 'race_white'], inplace=True) #TODO: re-construct these things 
    tract_demog['tract'] 
    tract_demog['tract'].apply(lambda x: x[0]).value_counts() 
    tract_demog['tract'].apply(len).value_counts() # all 10-digits that start with 6
    for vv in ['health_employer','health_medicare','health_medicaid','health_other']:
        tract_demog[vv] = tract_demog[vv].fillna(tract_demog[vv].mean())



tract_demog.loc[tract_demog['tr_pop'] == 0, 'tr_pop'] = 1 # avoid divide by zero
tracts = tracts.merge(tract_demog, on='tract', how='outer', indicator=True)
print("Merge to demographics:\n", tracts._merge.value_counts()) #perfect match
tracts = tracts.loc[tracts._merge != 'right_only', :]
tracts.drop(columns=['_merge'], inplace=True)
# drop tracts with zero population
tracts = tracts.loc[tracts['tr_pop'] > 0, :]

# TODO: impute health variables



# impute HPI for tracts with no HPI
impute_hpi_method = 'drop' # 'drop' or 'q1' or 'nearest'
if impute_hpi_method == 'drop':
    tracts = tracts.loc[tracts['hpi'].notnull(), :]
elif impute_hpi_method == 'q1':
    tracts.loc[tracts['hpi'].isnull(), 'hpi_quartile'] = 1 
    tracts.hpi_quartile.value_counts()
elif impute_hpi_method == 'nearest':
    #  TODO: get lat/lon of tracts to find
    tracts_tofind = tracts.loc[tracts['hpi'].isnull(), ['tract']]
    # make it the same as drop for now
    tracts = tracts.loc[tracts['hpi'].notnull(), :]



# merge with tract-level vote shares
tract_votes = pd.read_csv(f"{datadir}/Intermediate/tract_votes.csv", usecols=['tract', 'dshare'], dtype={'tract': str})
tract_votes['tract'].apply(len).value_counts() # 11
tract_votes['tract'].apply(lambda x: x[0]).value_counts() # all 0
tract_votes['tract'] = tract_votes['tract'].str[1:] # drop the first digit

tracts = tracts.merge(tract_votes, on='tract', how='outer', indicator=True)
print("Merge to votes:\n", tracts._merge.value_counts())
tracts = tracts.loc[tracts._merge != 'right_only', :]
tracts.drop(columns=['_merge'], inplace=True)
# impute vote shares
tracts['dshare'] = tracts['dshare'].fillna(tracts['dshare'].mean())


# merge with tract-ZIP crosswalk

tractzip_cw = pd.read_csv(f"{datadir}/Intermediate/tract_zip_crosswalk.csv", dtype={'zip': str, 'tract': str})
tractzip_cw['tract'] = tractzip_cw['tract'].str[1:] # drop the first digit
tractzip_cw['tract'].apply(len).value_counts() # all 10-digits that start with 6


agent_data = tracts.merge(tractzip_cw, on='tract', how='outer', indicator=True)
agent_data['_merge'].value_counts() # 5 left, 20k both
agent_data = agent_data.loc[agent_data._merge == 'both', :]
agent_data.drop(columns=['_merge'], inplace=True)

list(agent_data.columns)
agent_data.describe()

# Original ZIP-level data with vaccination rates (product_data) - should not be modified in this script
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv", dtype={'zip': str}) #from prep_demest.do
df['zip'].isin(agent_data['zip']).value_counts() #8 ZIPs in df but not in agent_data
df = df.assign(hpi_quartile = df['hpiquartile']) # for consistency with other quartile-based variables
# keep tracts that are in df
agent_data['zip'].isin(df['zip']).value_counts() # 19821 tracts in df, 205 not in df
agent_data = agent_data[agent_data['zip'].isin(df['zip'])]

# approximate race and income for ZIPs that double as its own tract
df = df.assign(race = 1-df.race_black-df.race_asian-df.race_hispanic-df.race_other)
df = df.assign(income = df.medianhhincome)


# weights to be the population of the tract over the population in the ZIP
agent_data = agent_data.merge(df[['zip', 'population']].rename(columns={'population':'zip_pop'}), on='zip', how='left')

pop_method = 'tract' # 'tract' or 'zip'
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

agent_data[['hpi']].describe()

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

agent_data = pd.concat([agent_data, aux_tracts], ignore_index=True)

agent_data['market_ids'] = agent_data['zip']
agent_data['nodes'] = 0 # for pyblp (no random coefficients)
agent_data['logdist'] = np.log(agent_data['dist']) 

print("agent_data.describe() \n", agent_data.describe())
print("product data describe() \n", df.describe())

# save to csv
agent_data.to_csv(f"{datadir}/Analysis/Demand/agent_data.csv", index=False)

# summarize tract distance
print("Distance (tract-level):")
print(agent_data['dist'].describe())
print("Log distance (tract-level):")
print(agent_data['logdist'].describe())
print("Distance (mean within ZIP):")
print(agent_data.groupby('market_ids')['dist'].mean().describe())
print("Log distance (mean within ZIP):")
print(agent_data.groupby('market_ids')['logdist'].mean().describe())
