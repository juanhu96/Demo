import pandas as pd

datadir = "/export/storage_covidvaccine/Data"


### Vaccination
vax_raw = pd.read_csv(f"{datadir}/Raw/Vaccination/covid19vaccinesbyzipcode_test.csv", dtype={'zip_code_tabulation_area': str})

vax_df = vax_raw.rename(columns={
    "as_of_date": "date",
    "zip_code_tabulation_area": "zip",
    "age12_plus_population": "pop12up",
    "tot_population": "population",
    "percent_of_population_fully_vaccinated": "vaxfull",
    "percent_of_population_partially_vaccinated": "vaxpart"
})

vax_df = vax_df[["date", "zip", "pop12up", "population", "vaxfull", "vaxpart"]]

vax_df.to_csv(f"{datadir}/Intermediate/vax_panel.csv", index=False)


#### HPI
hpi22 = pd.read_csv(f"{datadir}/Raw/HPI/hpi_zip_2022.csv", dtype={'geoid': str}, usecols=['geoid', 'percentile']).rename(columns={"geoid": "zip"})
hpi22 = hpi22.rename(columns={"percentile": "hpi_22"})

hpi11 = pd.read_csv(f"{datadir}/Raw/HPI/hpi_zip_2011.csv", dtype={'geoid': str}, usecols=['geoid', 'percentile']).rename(columns={"geoid": "zip"})
hpi11 = hpi11.rename(columns={"percentile": "hpi_11"})
# drop NAs 
hpi11 = hpi11.loc[~hpi11.hpi_11.isna(), :]

hpi_df = hpi22.merge(hpi11, on="zip", how="outer", indicator=True)
print(hpi_df._merge.value_counts()) #1313 both, 442 right_only, 0 left_only

hpi_df['hpi_percentile'] = hpi_df.hpi_22.fillna(hpi_df.hpi_11)
hpi_df.describe()
hpi_df = hpi_df[['zip', 'hpi_percentile']]

hpi_df.to_csv(f"{datadir}/Intermediate/hpi_zip.csv", index=False)


### ZIP demographics

