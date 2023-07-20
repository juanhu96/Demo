library(tidycensus)
library(tidyverse)
library(sf)


datadir <- "/export/storage_covidvaccine/Data/"

# v1 <- load_variables(2020, "acs5", cache = TRUE)
# View(v1)


# Load data from ACS
bg <- get_acs(
    year = 2020,
    geography = "block group",
    state = "CA",
    variables = "B01001_001",
    geometry = TRUE
) %>% select(GEOID, pop = estimate, geometry)


head(bg)

zips <- get_acs(
    year = 2020,
    geography = "zip code tabulation area",
    state = "CA",
    variables = "B01001_001",
    geometry = TRUE
) %>% select(GEOID, pop = estimate, geometry)

