# this will subsume prep_demest.do and trace back to raw data.

import numpy as np
import pandas as pd

datadir = "/export/storage_covidvaccine/Data"

###############
### Vaccination
###############
use_legacy_vax = True
if use_legacy_vax:
    vax_raw_path = f"{datadir}/Raw/Vaccination/covid19vaccinesbyzipcode_071222.csv"
else:
    vax_raw_path = f"{datadir}/Raw/Vaccination/covid19vaccinesbyzipcode_test.csv"

vax_raw = pd.read_csv(vax_raw_path)


vax_df = vax_raw.rename(columns={
    "as_of_date": "date",
    "zip_code_tabulation_area": "zip",
    "age12_plus_population": "population",
    "percent_of_population_fully_vaccinated": "vaxfull",
    "percent_of_population_partially_vaccinated": "vaxpart"
})

vax_df = vax_df[["date", "zip", "population", "vaxfull"]]
vax_df.sort_values(by=["date", "zip"], inplace=True)
vax_df.isna().sum()
# vax_df = vax_df.loc[~vax_df.vaxfull.isna(), :]

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
column_name_mapping = {
    'geoid': 'zip',
    # 'aland': 'landarea',
    'population': 'acs_population',
    'population_density_sq_mi': 'popdensity',
    'median_household_income': 'medianhhincome',
    'family_poverty_pct': 'poverty',
    'unemployment_pct': 'unemployment',
    'median_value_of_owner_occupied_units': 'medianhomevalue',
    'race_and_ethnicity_black': 'race_black',
    'race_and_ethnicity_asian': 'race_asian',
    'race_and_ethnicity_hispanic': 'race_hispanic',
    'race_and_ethnicity_other': 'race_other'
}

column_name_mapping.keys()
# Rename the columns in the California Demographics dataset
zip_demo = zip_demo_raw[column_name_mapping.keys()].rename(columns=column_name_mapping)

percap_vars = ['race_black','race_asian','race_hispanic','race_other']
for var in percap_vars:
    zip_demo[var] = zip_demo[var] / zip_demo['acs_population']

# drop acs_population as we're using the 12+ population from the vaccination data
zip_demo.drop(columns=['acs_population'], inplace=True)

zip_demo['medianhhincome'] = zip_demo['medianhhincome'] / 1000
zip_demo['medianhomevalue'] = zip_demo['medianhomevalue'] / 1000

zip_demo['zip'] = zip_demo['zip'].str.split('US').str[1]
zip_demo = zip_demo.loc[zip_demo['zip'] != '06']


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
zip_votes['dshare'].fillna(zip_votes['dshare'].mean(), inplace=True) #impute with mean



###############
# merge
###############

panel = vax_df.merge(hpi_df, on="zip", how="left", indicator=True)
print(panel._merge.value_counts()) 
# assign left_only to hpi=0
panel.loc[panel._merge == "left_only", "hpi"] = 0
# # inspect failed merges
# panel.loc[panel._merge == "left_only", :]
# vax_mar = vax_df.loc[vax_df.date == "2022-03-01", :]
# vax_mar.loc[~vax_mar.zip.isin(hpi_df.zip), :]
# vax_mar.merge(hpi_df, on="zip", how="left", indicator=True)._merge.value_counts()


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

###############

# compare with Analysis/Demand/demest_data.csv
# TODO: remove
demest_data = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv").reset_index(drop=True)


mar01 = pd.read_csv(f"{datadir}/Raw/notreallyraw/MAR01.csv").reset_index(drop=True)
mar01.columns.tolist()
mar01_tomerge = mar01[['Zip', 'VaxFull', 'Date']]

# CaliforniaVaccinationZip.csv
oldpanel = pd.read_csv(f"{datadir}/Raw/notreallyraw/CaliforniaVaccinationZip.csv").reset_index(drop=True)
oldpanel.columns.tolist()

vax_mar = vax_df.loc[vax_df.date == "2022-03-01", :].reset_index(drop=True)
vax_mar.isna().sum()

vax_mar_tomerge = vax_mar[['zip', 'vaxfull']]

vax_mar_tomerge.rename(columns={'zip': 'Zip', 'vaxfull': 'vaxfull_raw'}, inplace=True)
mar_compare = mar01_tomerge.rename(columns={'VaxFull': 'vaxfull_mar'}).merge(vax_mar_tomerge, on="Zip", how="outer", indicator=True)
mar_compare._merge.value_counts()
mar_compare.loc[mar_compare._merge == "right_only", :]
# display all
pd.set_option('display.max_rows', None)
mar_compare.loc[(mar_compare.vaxfull_mar.isna()) | (mar_compare.vaxfull_raw.isna()), :] #TODO:
mar_compare#TODO:


mar01.loc[vax_mar.vaxfull.isna(), ['Zip', 'VaxFull', 'Date']]
filled_zips = mar01.loc[vax_mar.vaxfull.isna(), ['Zip', 'VaxFull', 'Date']].Zip
vax_mar.loc[vax_mar.zip.isin(filled_zips), :]





demest_data.isna().sum()

df = panel.loc[panel.date == "2022-03-01", :]
# df = df.drop(columns=["date"])
# df = df.loc[df.zip.isin(demest_data.zip), :]


# set intersection of column names

for cc in set(df.columns) & set(demest_data.columns):
    print(round(np.mean(df[cc]), 3), round(np.mean(demest_data[cc]), 3), cc)

# inspect missings
df.isna().sum()
df.loc[df.vaxfull.isna(), :]
df = df.reset_index(drop=True)
demest_data.loc[df.vaxfull.isna(), :]


# TODO:  hpi, missing data, distance



##############
# prep_demest.do
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
cols_to_keep = ['zip', 'vaxfull', 'distnearest', 'hpi', 'hpiquartile', 'shares', 'race_black', 'race_asian', 
                'race_hispanic', 'race_other', 'health_employer', 'health_medicare', 'health_medicaid', 
                'health_other', 'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
                'medianhomevalue', 'popdensity', 'population']
df = df[cols_to_keep]



# df.rename(columns={'distnearest': 'dist'}, inplace=True)
# df['logdist'] = np.log(df['dist'])

df['market_ids'] = df['zip']
df['firm_ids'] = 1
df['prices'] = 0



# set difference of column names
set(df.columns) - set(demest_data.columns)
set(demest_data.columns) - set(df.columns)
# intersection of column names
set(df.columns) & set(demest_data.columns)



df_hpi = pd.read_csv(f'{datadir}/Raw/hpi2score_zip_2011.csv')
df_hpi.rename(columns={'geoid': 'zip', 'value': 'hpi_val', 'percentile': 'hpi'}, inplace=True)
df = pd.merge(df, df_hpi, on='zip', how='left')

# Save final data
df.to_csv(f'{datadir}/Analysis/Demand/demest_data.csv', index=False)
