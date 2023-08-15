# read in block coordinates and population, save as blk_coords_pop.csv

import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Data/"

# read in block coordinates and population, save as blk_coords_pop.csv
# source: https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/California/
# after download, unzip, change extension from .pl to .csv, and save as utf-8

t1 = pd.read_csv(f"{datadir}/Raw/blocks/ca000012020.csv", sep='|', header=None)
tg = pd.read_csv(f"{datadir}/Raw/blocks/cageo2020.csv", sep='|', header=None, encoding='utf-8')

tg_lvl = tg[2]
sum(tg_lvl == 750) #block level code
tgblk = tg.loc[tg_lvl == 750]
tg[[7]].describe() #logrecno
tg[[92]].describe() #lat
tg[[93]].describe() #long
tgsub = tgblk.loc[:, [7,92,93]] # 92: lat, 93: long

t1[[4]].describe() #logrecno
t1[[5]].describe() #population
t1sub = t1.loc[:,4:5]

print(t1sub.head())
print(tgsub.head())

t1sub.columns = ['logrecno', 'population']
tgsub.columns = ['logrecno', 'lat', 'long']

# merge
blk = pd.merge(tgsub, t1sub, on='logrecno', how='outer', indicator=True)
blk.head()
blk.shape
blk['_merge'].value_counts() #some right_only - other levels of geography
blk = blk.loc[blk['_merge'] == 'both']

blk['population'].describe()
blk['population'].sum() #equal to tg.loc[0,90] 
np.mean(blk['population'] == 0) # 27% of blocks have 0 population
blk = blk.loc[blk['population'] > 0]

blk = blk[['lat', 'long', 'population']]
blk.shape

# assign a blkid for this project
blk['blkid'] = blk.index

blk.to_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", index=False)




