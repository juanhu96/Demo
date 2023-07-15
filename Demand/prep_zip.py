# this subsumes prep_demest.do and trace back to raw data.

import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Data"

###############
### Vaccination
###############

vax_raw_path = f"{datadir}/Raw/Vaccination/covid19vaccinesbyzipcode_071222.csv" #legacy table with population age 5+
vax_raw = pd.read_csv(vax_raw_path)
vax_raw.columns.tolist()

vax_rename = {
    "as_of_date": "date",
    "zip_code_tabulation_area": "zip",
    "age12_plus_population": "pop12up",
    "age5_plus_population": "pop5up",
    "percent_of_population_fully_vaccinated": "vaxfull",
    "percent_of_population_partially_vaccinated": "vaxpart"
}
vax_df = vax_raw[vax_rename.keys()].rename(columns=vax_rename)

vax_df.sort_values(by=["date", "zip"], inplace=True)

# Interpolate missing vaxfull values (12194)
# set index for interpolation
vax_df['date'] = pd.to_datetime(vax_df['date'])
vax_df.set_index(['zip', 'date'], inplace=True)

# interpolate missing vaxfull with linear interpolation (12194 -> 7585 missing)
vax_zipgrp = vax_df.groupby(level=0)['vaxfull']
vax_df['vaxfull'] = vax_zipgrp.apply(lambda group: group.interpolate(method='linear'))

# front-fill missing vaxfull with last observation (7585 -> 3 missing)
vax_df['vaxfull'] = vax_df['vaxfull'].fillna(method='ffill') 

# back-fill missing vaxfull with first observation (3 -> 0 missing)
vax_df['vaxfull'] = vax_df['vaxfull'].fillna(method='bfill')

vax_df.reset_index(inplace=True)
vax_df.to_csv(f"{datadir}/Intermediate/vax_panel.csv", index=False)

###############
### HPI
###############
hpi22 = pd.read_csv(f"{datadir}/Raw/HPI/hpi_zip_2022.csv", usecols=['geoid', 'percentile']).rename(columns={"geoid": "zip"})
hpi22 = hpi22.rename(columns={"percentile": "hpi_22"})

hpi11 = pd.read_csv(f"{datadir}/Raw/HPI/hpi_zip_2011.csv", usecols=['geoid', 'percentile']).rename(columns={"geoid": "zip"})
hpi11 = hpi11.rename(columns={"percentile": "hpi_11"})
# drop NAs 
hpi11 = hpi11.loc[~hpi11.hpi_11.isna(), :]

hpi_df = hpi22.merge(hpi11, on="zip", how="outer", indicator=True)
print(hpi_df._merge.value_counts()) #1313 both, 442 right_only, 0 left_only

hpi_df['hpi'] = hpi_df.hpi_22.fillna(hpi_df.hpi_11)
hpi_df = hpi_df[['zip', 'hpi']]


# hpi_df.loc[~hpi_df.zip.isin(vax_df.zip), :]
# vax_df.loc[~vax_df.zip.isin(hpi_df.zip), :]

hpi_df.to_csv(f"{datadir}/Intermediate/hpi_zip.csv", index=False)


###############
### demographics
###############
demog_path = "California_DemographicsByZip2020.xlsx" #from Cubit
zip_demo_raw = pd.read_excel(f"{datadir}/Raw/{demog_path}", header=4)
zip_demo_raw.columns.tolist()
# Create a dictionary for mapping the column names in the California Demographics dataset to the equivalent column names in the MAR01.csv file
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
# Rename the columns in the California Demographics dataset
zip_demo = zip_demo[column_name_mapping.keys()].rename(columns=column_name_mapping)

# remove row with entire US 
zip_demo = zip_demo.loc[zip_demo['zip'] != '06']

percap_vars = ['race_black','race_asian','race_hispanic','race_other']
for var in percap_vars:
    zip_demo[var] = zip_demo[var] / zip_demo['acs_population']

zip_demo['medianhhincome'] = zip_demo['medianhhincome'] / 1000
zip_demo['medianhomevalue'] = zip_demo['medianhomevalue'] / 1000


zip_demo['zip'] = zip_demo['zip'].str.split('US').str[1]


zip_demo.to_csv(f"{datadir}/Intermediate/zip_demo.csv", index=False)


###############
# read
###############
vax_df = pd.read_csv(f"{datadir}/Intermediate/vax_panel.csv")
hpi_df = pd.read_csv(f"{datadir}/Intermediate/hpi_zip.csv")
zip_demo = pd.read_csv(f"{datadir}/Intermediate/zip_demo.csv")

# health insurance
zip_health = pd.read_csv(f"{datadir}/Intermediate/zip_health.csv") # from zip_health.R

# voting shares
zip_votes = pd.read_csv(f'{datadir}/Intermediate/zip_votes.csv', usecols=['zip', 'dshare']) # from voteshares.py


###############
# impute demog and health variables with mean TODO: check
###############

for vv in ["race_black", "race_asian", "race_hispanic", "race_other"]:
    zip_demo[vv] = zip_demo[vv].fillna(np.mean(zip_demo[vv]))

for vv in ['health_employer', 'health_medicare', 'health_medicaid', 'health_other', 'health_none']:
    zip_health[vv] = zip_health[vv].fillna(np.mean(zip_health[vv]))

zip_votes['dshare'] = zip_votes['dshare'].fillna(np.mean(zip_votes['dshare']))

###############
# merge
###############

panel = vax_df.merge(hpi_df, on="zip", how="left", indicator=True)
print(panel._merge.value_counts()) 
# assign left_only to hpi=0, per HPI documentation
panel.loc[panel._merge == "left_only", "hpi"] = 0


panel.drop(columns=["_merge"], inplace=True)


panel = panel.merge(zip_demo, on="zip", how="left", indicator=True)
print(panel._merge.value_counts())
panel.drop(columns=["_merge"], inplace=True)

panel = panel.merge(zip_health, on="zip", how="left", indicator=True)
print(panel._merge.value_counts())
panel.drop(columns=["_merge"], inplace=True)

panel = panel.merge(zip_votes, on="zip", how="left", indicator=True)
print(panel._merge.value_counts()) #some missing votes, can leave for now
panel.drop(columns=["_merge"], inplace=True)

# TODO: decide on ZIP population - using 5+ population from vaccination data for now
panel['population'] = panel['pop5up']


###############

# old datasets to compare to

# demest_data = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv").reset_index(drop=True)

# mar01 = pd.read_csv(f"{datadir}/Raw/notreallyraw/MAR01.csv").reset_index(drop=True)

# oldpanel = pd.read_csv(f"{datadir}/Raw/notreallyraw/CaliforniaVaccinationZip.csv").reset_index(drop=True)


# for cc in set(df.columns) & set(demest_data.columns):
#     print(round(np.mean(df[cc]), 3), round(np.mean(demest_data[cc]), 3), cc)



###############
# continue pipeline
###############

# subset to March 2022
df = panel.loc[panel.date == "2022-03-01", :]


##############
# analog to prep_demest.do: 
##############

df['shares'] = df['vaxfull']
df['shares'] = df['shares'].clip(0.05, 0.95)

prep_logit = False
if prep_logit:
    df['shares_out'] = 1 - df['shares']
    df['logshares'] = np.log(df['shares'])
    df['logshares_out'] = np.log(df['shares_out'])
    df['delta'] = df['logshares'] - df['logshares_out']


# Subset data
cols_to_keep = ['zip', 'vaxfull', 'hpi', 'shares', 'race_black', 'race_asian', 
                'race_hispanic', 'race_other', 'health_employer', 'health_medicare', 'health_medicaid', 'dshare',
                'health_other', 'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
                'medianhomevalue', 'popdensity', 'population']
df = df[cols_to_keep]


df['market_ids'] = df['zip']
df['firm_ids'] = 1
df['prices'] = 0

# Will compute distances (for ZIPs that need it) later
# Will compute HPI quantiles later

# inspect 
pd.options.display.max_columns = None
print(df.describe())
print(df.isna().sum())


# Save final data
df.to_csv(f'{datadir}/Analysis/Demand/demest_data.csv', index=False)

# read
df = pd.read_csv(f'{datadir}/Analysis/Demand/demest_data.csv', dtype={'zip': str})
