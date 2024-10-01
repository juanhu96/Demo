#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sep, 2024 by Jingyuan Hu
"""

# this subsumes prep_demest.do and trace back to raw data.
# this is to be run after prep_block.py

import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Demo/Data"

df = pd.read_csv(f"{datadir}/../areas.csv")
df.rename(columns={'ZIP': 'zip'}, inplace=True)

##############
# analog to prep_demest.do: 
##############

df['shares'] = df['Rates']
# subject to 5% and 95% winsorization
df['shares'] = df['shares'].clip(0.05, 0.95) 

block_data = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
block_data = block_data.loc[block_data['zip'].isin(df['zip']), :]
block_data.to_csv(f"{datadir}/Analysis/Demand/agent_data.csv", index=False)
cw_pop = block_data[["blkid", "market_ids", "population"]]
cw_pop.sort_values(by=['blkid'], inplace=True)
cw_pop.to_csv(f"{datadir}/Analysis/Demand/cw_pop.csv", index=False)

df = df.loc[df['zip'].isin(block_data['zip']), :]

# for pyblp
df['market_ids'] = df['zip']
df['firm_ids'] = 1
df['prices'] = 0

# Will compute distances (for ZIPs that need it) later
# Will compute HPI quantiles later

# Save final data
df.to_csv(f'{datadir}/Analysis/Demand/demest_data.csv', index=False)
