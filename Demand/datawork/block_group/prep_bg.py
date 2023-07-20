# prepare block group data for demand estimation

import pandas as pd
import numpy as np

datadir = "/export/storage_covidvaccine/Data"

pharmdist = pd.read_csv(f"{datadir}/Intermediate/ca_bg_pharm_dist.csv", dtype={'id': str})

cw = pd.read_csv(f"{datadir}/Intermediate/bg_zip_crosswalk.csv", dtype={'bg': str, 'zip': str})

bgpop = pd.read_csv(f"{datadir}/Intermediate/bg_pop.csv", dtype={'bg': str})


# number of block groups per zip
cw.groupby('zip').size().describe()

# number of zips per block group
cw.groupby('bg').size().describe()
cw.groupby('bg').size().value_counts().sort_index()/len(set(cw['bg']))

