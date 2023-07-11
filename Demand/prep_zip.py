import pandas as pd

datadir = "/export/storage_covidvaccine/Data"

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

#######investigate HPI
# merge in HPI
hpi22 = pd.read_csv(f"{datadir}/Raw/hpi_zip_2022.csv", dtype={'geoid': str}, usecols=['geoid', 'value', 'percentile']).rename(columns={"geoid": "zip", "value": "hpi"})
hpi11 = pd.read_csv(f"{datadir}/Raw/hpi_zip_2011.csv", dtype={'geoid': str}, usecols=['geoid', 'value', 'percentile']).rename(columns={"geoid": "zip", "value": "hpi"})
hpi11 = hpi11.rename(columns={"percentile": "percentile_11", "hpi": "hpi_11"})
hpi22 = hpi22.rename(columns={"percentile": "percentile_22", "hpi": "hpi_22"})

hpi11.isna().sum() # 12 out of 1767 missing
hpi11.loc[hpi11.zip == "90263"]
hpi11.loc[hpi11.zip == "90743"]


hpi_compare = hpi22.merge(hpi11, on="zip", how="outer", indicator=True)
hpi_compare._merge.value_counts()
hpi_compare.loc[hpi_compare._merge == "right_only", :] #454 rows
hpi_compare.loc[hpi_compare._merge == "right_only", :].isna().sum() # 454 rows

hpi_compare['hpi_filled'] = hpi_compare.hpi_22.fillna(hpi_compare.hpi_11)
# compute quantile of hpi_filled
hpi_compare['percentile_filled'] = hpi_compare.hpi_filled.rank(pct=True)

hpi_compare.loc[hpi_compare.hpi_22.isna(), :]




mar_df = pd.read_csv(f"{datadir}/Raw/notreallyraw/MAR01.csv", dtype={'Zip': str}, usecols=['Zip', 'HPI'])
mar_df = mar_df.rename(columns={"Zip": "zip", "HPI": "percentile_mar"})


mar_compare = mar_df.merge(hpi11, on="zip", how="outer", indicator=True)
mar_compare = mar_compare.rename(columns={"_merge": "merge_11"}) 
mar_compare.merge_11.value_counts() # 1750 both, 17 right_only, 0 left_only
mar_compare.loc[mar_compare.merge_11 == "both",:]
# "hpi" in mar_df (i.e. hpi_x in mar_compare) is really the percentile. 

# display all
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

mar_compare = mar_compare.merge(hpi22, on="zip", how="outer", indicator=True)
mar_compare = mar_compare.rename(columns={"_merge": "merge_22"})
mar_compare.merge_22.value_counts() # 1313 both , 454 left_only
mar_compare.sort_values(by="zip", inplace=True)
mar_compare = mar_compare

mar_compare.loc[mar_compare.percentile_mar.isna(), :]
mar_compare.loc[mar_compare.percentile_11.isna(), :]
mar_compare.loc[mar_compare.percentile_22.isna(), :] # 454 rows. This is mysterious.
mar_compare.loc[mar_compare.percentile_22.isna(), :].isna().sum() 


# inspect zip 96142 for all files
mar_df.loc[mar_df.zip == "96142", :]
hpi11.loc[hpi11.zip == "96142", :]
hpi22.loc[hpi22.zip == "96142", :]
mar_compare.loc[mar_compare.zip == "96142", :]
hpi_compare.loc[hpi_compare.zip == "96142", :]

hpi11.isna().sum() # 12 out of 1767 missing
hpi22.isna().sum() # 0 out of 1313 missing
mar_df.isna().sum() # 0 out of 1750 missing

# HPI non-missing in MAR but missing in HPI22
mar_compare.loc[(mar_compare.percentile_mar.notna()) & (mar_compare.percentile_22.isna()), :].isna().sum() # 454 rows

# HPI non-missing in MAR but missing in HPI11
mar_compare.loc[(mar_compare.percentile_mar.notna()) & (mar_compare.percentile_11.isna()), :] #none

# HPI missing in MAR but non-missing in HPI11
mar_compare.loc[(mar_compare.percentile_mar.isna()) & (mar_compare.percentile_11.notna()), :].shape #none

hpi11.zip.nunique() # 1767
hpi22.zip.nunique() # 1313


#######investigate TRACT HPI
