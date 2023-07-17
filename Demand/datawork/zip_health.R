library(tidycensus)
library(tidyverse)

datadir <- "/export/storage_covidvaccine/Data/"

# Load data from ACS
acs_raw <- get_acs(
    year = 2019,
    geography = "zip code tabulation area",
    state = "CA",
    table = "B27010"
)

acs_raw <- acs_raw[c("GEOID", "variable", "estimate")]


zip_health <- acs_raw %>%
  spread(key = "variable", value = "estimate")

zip_health_out <- zip_health %>%
  mutate(population = B27010_001) %>%
  mutate(health_employer = (B27010_004 + B27010_011 + B27010_020 + B27010_027 + B27010_036 + B27010_043 + B27010_053 + B27010_059) / population) %>%
  mutate(health_medicare = (B27010_006 + B27010_012 + B27010_022 + B27010_028 + B27010_038 + B27010_044 + B27010_045 + B27010_055 + B27010_060 + B27010_061 + B27010_062) / population) %>%
  mutate(health_medicaid = (B27010_007 + B27010_013 + B27010_023 + B27010_029 + B27010_039 + B27010_046) / population) %>%
  mutate(health_other = (B27010_005 + B27010_008 + B27010_009 + B27010_014 + B27010_015 + B27010_016 + B27010_021 + B27010_024 + B27010_025 + B27010_030 + B27010_031 + B27010_032 + B27010_037 + B27010_040 + B27010_041 + B27010_047 + B27010_048 + B27010_049 + B27010_054 + B27010_056 + B27010_057 + B27010_063 + B27010_064 + B27010_065) / population) %>%
  mutate(health_none = (B27010_017 + B27010_033 + B27010_050 + B27010_066) / population)

# Drop columns
zip_health_out <- zip_health_out %>% 
  select(-starts_with("B27010")) %>% 
  select(-population)


# Rename columns
zip_health_out <- zip_health_out %>% rename(zip = GEOID)

# Export
write.csv(zip_health_out, file = paste0(datadir, "Intermediate/zip_health.csv"), row.names = FALSE)
