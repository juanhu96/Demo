#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sep, 2024 by Jingyuan Hu
"""

# read in block coordinates and population, save as blk_coords_pop.csv

import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Demo/Data"

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
tgsub = tgblk.loc[:, [7,8,92,93]] # 92: lat, 93: long
tgsub[[8]].head()

t1[[4]].describe() #logrecno
t1[[5]].describe() #population
t1sub = t1.loc[:,4:5]

print(t1sub.head())
print(tgsub.head())

t1sub.columns = ['logrecno', 'population']
tgsub.columns = ['logrecno', 'geoid', 'lat', 'long']

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

blk = blk[['lat', 'long', 'population', 'geoid']]
blk.shape

# assign a blkid for this project
blk['blkid'] = blk.index

# correcting population to 5+, integrate into above code

blkpop5 = pd.read_csv(f"{datadir}/Raw/blocks/DECENNIALDHC2020.P12_2023-09-18T193028/DECENNIALDHC2020.P12-Data.csv", sep=',', usecols=[0,2,6,54], skiprows=1)
blkpop5.columns = ['geoid', 'total', 'popunder5m', 'popunder5f']
blkpop5['total'].sum()
blkpop5['pop5plus'] = blkpop5['total'] - blkpop5['popunder5m'] - blkpop5['popunder5f']
blkpop5 = blkpop5[['geoid', 'pop5plus']]
blkpop5['geoid'] = blkpop5['geoid'].str.split('US').str[1].astype(int)
# merge

blk['geoid'] = blk['geoid'].str.split('US').str[1].astype(int)
blk_m = pd.merge(blk, blkpop5, on='geoid', how='outer', indicator=True)
blk_m.head()
blk_m.shape
blk_m['_merge'].value_counts() #some right_only 
blk_m = blk_m.loc[blk_m['_merge'] == 'both']

blk_m['pop5plus'].describe()
np.corrcoef(blk_m['population'], blk_m['pop5plus']) 
blk_m['pop5plus'].sum()
blk_m['population'] = blk_m['pop5plus']
blk_m = blk_m[['lat', 'long', 'population', 'blkid']]
blk_m = blk_m.loc[blk_m['population'] > 0]

blk_m.to_csv(f"{datadir}/Intermediate/blk_coords_pop.csv", index=False)