library(tidyverse)

datadir <- "/export/storage_covidvaccine/Data"
TRACT = read.csv(paste0(datadir, "/Raw/ACS/ACSDT5Y2019.B27010.csv"))

HEALTH = TRACT %>%
  mutate(Population = B27010_001E) %>%
  mutate(Health_Employer = (B27010_004E + B27010_011E + B27010_020E + B27010_027E + B27010_036E + B27010_043E + B27010_053E + B27010_059E) / Population) %>%
  mutate(Health_Medicare = (B27010_006E + B27010_012E + B27010_022E + B27010_028E + B27010_038E + B27010_044E + B27010_045E + B27010_055E + B27010_060E + B27010_061E + B27010_062E) / Population) %>%
  mutate(Health_Medicaid = (B27010_007E + B27010_013E + B27010_023E + B27010_029E + B27010_039E + B27010_046E) / Population) %>%
  mutate(Health_Other = (B27010_005E + B27010_008E + B27010_009E + B27010_014E + B27010_015E + B27010_016E + B27010_021E + B27010_024E + B27010_025E + B27010_030E + B27010_031E + B27010_032E + B27010_037E + B27010_040E + B27010_041E + B27010_047E + B27010_048E + B27010_049E + B27010_054E + B27010_056E + B27010_057E + B27010_063E + B27010_064E + B27010_065E) / Population) %>%
  mutate(Health_None = (B27010_017E + B27010_033E + B27010_050E + B27010_066E) / Population) %>%
  mutate(Total = Health_Employer + Health_Medicare + Health_Medicaid + Health_Other + Health_None)

HEALTH_CA = HEALTH %>%
  mutate(geoid = substr(GEO_ID, 11, 20), tract = as.numeric(geoid), State = substr(GEO_ID, 10, 11)) %>%
  filter(State == "06") %>%
  select(c(tract, Health_Employer, Health_Medicare, Health_Medicaid, Health_Other, Health_None)) %>%
  distinct()

# TODO: where does the original TRACT_merged come from?

# DEMO = read.csv("C:/Users/elong/Dropbox/COVID Vaccination/Vaccination Data/California/TRACT_merged.csv")

# TRACT_merged = left_join(DEMO, HEALTH_CA, by = "tract")

# write.csv(TRACT_merged, "C:/Users/elong/Dropbox/COVID Vaccination/Vaccination Data/California/TRACT_merged.csv")
