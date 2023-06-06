import pandas as pd
import pyblp
import numpy as np
poolnum = 32
pyblp.options.digits = 4

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data

# Original ZIP-level data with vaccination rates
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0
df.zip = df.zip.astype(str)
list(df.columns)


# Tract (geoid) level data with HPI and HPIQuartile
tract_hpi_df = pd.read_csv(f"{datadir}/tracts/HPItract.csv")
tract_hpi_df


# Tract-ZIP crosswalk
tractzip_df = pd.read_csv(f"{datadir}/tracts/HPITractZip.csv", usecols=['Tract', 'Zip'], dtype={'Tract': str, 'Zip': str})
tractzip_df.shape
tractzip_df
# drop if Zip is NA
tractzip_df = tractzip_df.dropna(subset=['Zip'])
tractzip_df.shape
tractzip_df = tractzip_df.sort_values(by=['Zip'])

# keep if Zip is in df
tractzip_df = tractzip_df[tractzip_df['Zip'].isin(df['zip'])]
tractzip_df.shape

tractzip_df['Zip'].unique().shape
len(np.unique(tractzip_df['Tract']))
tractzip_df['Tract'].apply(len).value_counts() # all 10

correcttracts = sorted(np.unique(tractzip_df['Tract']))
correcttracts[:10]
correcttracts[-30:]



# This is some tract-ZIP level stuff that has way too many ZIPs
tract_df = pd.read_csv(f"{datadir}/tracts/tract_zip_032022.csv", usecols=['TRACT', 'ZIP'], dtype={'TRACT': str, 'ZIP': str})
# verify that TRACT is 11 digits
tract_df['TRACT'].apply(len).value_counts() # all 11
tract_df = tract_df.assign(statefips = tract_df['TRACT'].str[:3])
# keep CA only
tract_df = tract_df[tract_df['statefips'] == '060']
tract_df = tract_df.assign(countyfips = tract_df['TRACT'].str[3:6])
tract_df['countyfips'] = tract_df['countyfips'].apply(lambda x: x.zfill(4))

tract_df = tract_df.assign(tractid = tract_df['TRACT'].str[6:])
# Make the Tract column the same as the one in tract_nearest_df
tract_df = tract_df.assign(Tract = '6' + tract_df['countyfips'] + tract_df['tractid'])



# 2019 ACS tract demographics (has a ton of variables)
# tract_demog_df = pd.read_csv(f"{datadir}/tracts/CA_TRACT_demographics.csv", low_memory=False)



# read tract-pharmacy distance pairs
# pre-processed to the 10 nearest pharmacies for each tract.
pairs_df = pd.read_csv(f"{datadir}/tracts/pairs_filtered.csv", usecols=['Tract', 'CA_Pharm_ID', 'Distance'],dtype={'Tract': str, 'CA_Pharm_ID': str, 'Distance': float})
pairs_df
# just the nearest pharmacy for each tract
tract_nearest_df = pairs_df.groupby('Tract').head(1)
tract_nearest_df.sort_values(by=['Tract']).iloc[-30:,]
tract_nearest_df['Tract'].apply(len).value_counts() # between 8 and 11
# The tract column is messed up. I think there should be county FIPS as the first 5. Followed by a 5 digit tract ID. 
# TODO: verify if my fix is correct
# look at some examples 
tract_nearest_df.groupby(tract_nearest_df['Tract'].apply(len)).apply(lambda x: x.sample(10))
# check that the first digit is always 0
tract_nearest_df['Tract'].apply(lambda x: x[0]).value_counts()
tract_nearest_df = tract_nearest_df.assign(countyfips = tract_nearest_df['Tract'].str[1:6])
tract_nearest_df['countyfips']
tract_nearest_df['tractid'] = tract_nearest_df['Tract'].str[6:]
tract_nearest_df['tractid']
# pad the tractid with 0s
tract_nearest_df['tractid'] = tract_nearest_df['tractid'].apply(lambda x: x.zfill(5))
# combine the countyfips and tractid
tract_nearest_df['Tract'] = tract_nearest_df['countyfips'] + tract_nearest_df['tractid']


# merge tract_nearest_df with tractzip_df 

agent_data = tract_nearest_df.merge(tractzip_df, on='Tract', how='outer', indicator=True)
agent_data._merge.value_counts()
agent_data = agent_data[agent_data['_merge'] == 'both']
# agent_data.drop(columns=['_merge', 'countyfips', 'tractid', 'CA_Pharm_ID'], inplace=True)
agent_data.sort_values(by=['Zip'], inplace=True)
agent_data
agent_data['Tract'].unique().shape
agent_data['Zip'].unique().shape


# see if there's zips in df that aren't in agent_data
df['zip'].isin(agent_data['Zip']).value_counts()
df['zip']
agent_data['Zip']


# merge tract_nearest_df with tract_df
tract_nearest_df['Tract']
tract_df.Tract
merged = tract_df.merge(tract_nearest_df, on='Tract', how='outer', indicator=True)
"merged"._merge.value_counts()

# see if there's zips in df that aren't in merged
df['zip'].isin(merged['ZIP']).value_counts()


















