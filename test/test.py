#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sep, 2024 by Jingyuan Hu
"""

# this subsumes prep_demest.do and trace back to raw data.
# this is to be run after prep_block.py

import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Data" # to get sample areas only

ZIP_VAX = pd.read_csv(f"{datadir}/Intermediate/vax_panel.csv", usecols=['date', 'zip', 'vaxfull'], dtype={'zip': str})
ZIP_VAX = ZIP_VAX.loc[ZIP_VAX.date == "2022-03-01", :].copy()

ZIP = pd.read_csv(f'{datadir}/Raw/ZIP_Code_Population_Weighted_Centroids.csv', usecols=['STD_ZIP5', 'USPS_ZIP_PREF_STATE_1221', 'LATITUDE', 'LONGITUDE'], dtype={'STD_ZIP5': str})
ZIP = ZIP.rename(columns={'STD_ZIP5': 'zip', 'USPS_ZIP_PREF_STATE_1221': 'STATE'})
ZIP = ZIP[ZIP['STATE'] == 'CA'].copy()
ZIP = ZIP.merge(ZIP_VAX, on='zip', how='inner')

# ============================= DEMOGRAPHICS ==================================

zip_demo_raw = pd.read_excel(f"{datadir}/Raw/California_DemographicsByZip2020.xlsx", header=4)
zip_demo = zip_demo_raw.assign(
    collegegrad = zip_demo_raw['educational_attainment_bachelors'] + zip_demo_raw['educational_attainment_graduate'])

column_name_mapping = {
    'geoid': 'zip',
    # 'aland': 'landarea',
    'population': 'acs_population',
    'population_density_sq_mi': 'popdensity',
    'median_household_income': 'medianhhincome',
    'family_poverty_pct': 'poverty',
    'unemployment_pct': 'unemployment',
    'median_value_of_owner_occupied_units': 'medianhomevalue',
    'collegegrad': 'collegegrad',
    'race_and_ethnicity_black': 'race_black',
    'race_and_ethnicity_asian': 'race_asian',
    'race_and_ethnicity_hispanic': 'race_hispanic',
    'race_and_ethnicity_other': 'race_other'
}

column_name_mapping.keys()
zip_demo = zip_demo[column_name_mapping.keys()].rename(columns=column_name_mapping) 
zip_demo = zip_demo.loc[zip_demo['zip'] != '06']

percap_vars = ['race_black','race_asian','race_hispanic','race_other']
for var in percap_vars:
    zip_demo[var] = zip_demo[var] / zip_demo['acs_population']

zip_demo['medianhhincome'] = zip_demo['medianhhincome'] / 1000
zip_demo['medianhomevalue'] = zip_demo['medianhomevalue'] / 1000

zip_demo['zip'] = zip_demo['zip'].str.split('US').str[1]

ZIP = ZIP.merge(zip_demo, on='zip', how='inner')

# ============================= HPI ==================================

hpi22 = pd.read_csv(f"{datadir}/Raw/HPI/hpi_zip_2022.csv", usecols=['geoid', 'percentile']).rename(columns={"geoid": "zip"})
hpi22 = hpi22.rename(columns={"percentile": "hpi_22"})
hpi11 = pd.read_csv(f"{datadir}/Raw/HPI/hpi_zip_2011.csv", usecols=['geoid', 'percentile']).rename(columns={"geoid": "zip"})
hpi11 = hpi11.rename(columns={"percentile": "hpi_11"})
hpi11 = hpi11.loc[~hpi11.hpi_11.isna(), :]
hpi_df = hpi22.merge(hpi11, on="zip", how="outer", indicator=True)
hpi_df['hpi'] = hpi_df.hpi_22.fillna(hpi_df.hpi_11)
hpi_df = hpi_df[['zip', 'hpi']]
hpi_df['hpi_quartile'] = pd.cut(hpi_df['hpi'], 4, labels=False, include_lowest=True) + 1

hpi_df['zip'] = hpi_df['zip'].astype(str)
ZIP['zip'] = ZIP['zip'].astype(str)

ZIP = ZIP.merge(hpi_df, on="zip", how="inner")

# ============================= DEMAND ==================================

zip_codes_to_filter = ['90059', '90201', '90277', '90290', '90302', '90503', '90605', '90606', '90703', '90732']
areas = ZIP[ZIP['zip'].isin(zip_codes_to_filter)]

# areas = areas[['zip', 'LATITUDE', 'LONGITUDE', 'vaxfull', 'hpi', 'hpi_quartile', 
# 'acs_population', 'popdensity', 'medianhhincome', 'poverty', 'unemployment', 'medianhomevalue', 'collegegrad']]
areas = areas[['zip', 'LATITUDE', 'LONGITUDE', 'vaxfull', 'hpi', 'hpi_quartile', 'unemployment', 'collegegrad']]
areas.to_csv(f"{datadir}/../Demo/areas_new.csv", index=False)