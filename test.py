#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import pandas as pd
import numpy as np

datadir='/export/storage_covidvaccine/Data/'
resultdir='/export/storage_covidvaccine/Result/'

distdf_pharmacy = pd.read_csv(f'{datadir}/Intermediate/ca_blk_pharm_dist.csv', dtype={'locid': int, 'blkid': int})
distdf_dollar = pd.read_csv(f'{resultdir}MaxVaxHPIDistBLP/M5_K8000/Dollar/vaccinated/ca_blk_Dollar_dist_total.csv', dtype={'locid': int, 'blkid': int})
distdf_chain = pd.read_csv(f'{datadir}/Intermediate/ca_blk_Dollar_dist.csv', dtype={'locid': int, 'blkid': int})
print(np.max(distdf_chain.locid))

distdf_chain.locid = distdf_chain.locid + 4035

print(distdf_pharmacy.head)
print(distdf_dollar.head)
print(distdf_chain.head)
