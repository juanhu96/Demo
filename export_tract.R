# Geographic ID (tract or zip)
# Lat/Long of population centroid
# 2020 election results (2-party vote share won by Biden), imputed from precinct to tract or zip
# HPI and HPI quartile
# ACS demographics: 
#   Population
#   Population density ---> LAND_AREA (in square miles)
#   Race (% white, black, asian, hispanic, other)
#   Health insurance (% employer, medicare, medicaid, other) ---> not available, (One_Health_Ins_ACS_14_18, Two_Plus_Health_Ins_ACS_14_18, No_Health_Ins_ACS_14_18)
#   College graduation rate ---> pct_College_ACS_14_18
#   Unemployment rate ---> pct_Civ_unemp_16p_ACS_14_18
#   Poverty level ---> pct_Prs_Blw_Pov_Lev_ACS_14_18
#   Median household income ---> Med_HHD_Inc_ACS_14_18
#   Median home value ---> Med_House_Value_ACS_14_18

################################################################################

library(dplyr)
library(janitor)
setwd("/export/storage_covidvaccine/")

################################################################################
### 1. Tract level ###


## Import ##
TRACT_VOTES <- read.csv("Data/Intermediate/tract_votes.csv", stringsAsFactors = FALSE) # from voteshares.py
TRACT_HPI <- read.csv("Data/Raw/HPI/hpi_tract_2022.csv", stringsAsFactors = FALSE) # downloaded
POPULATION_CENTROIDS <- read.csv("Data/Intermediate/tract_centroids.csv", stringsAsFactors = FALSE) # from process_raw_data.R

DEMOGRAPHICS <- read.csv("Data/Raw/Census/pdb2020trv2_us.csv")
DEMOGRAPHICS <- DEMOGRAPHICS %>% filter(State == 6)
# DEMOGRAPHICS <- read.csv("Data/Tract_demographics.csv", stringsAsFactors = FALSE)


## Rename ##
TRACT_HPI <- TRACT_HPI %>% rename("tract" = "geoid")
POPULATION_CENTROIDS <- POPULATION_CENTROIDS %>% rename("tract" = "GEOID") %>% select(-c(ID))
# Use Census 2010 since it also matches the population in POPULATION_CENTROIDS
DEMOGRAPHICS <- DEMOGRAPHICS %>% dplyr::select(c(GIDTR, LAND_AREA,
                                                 Tot_Population_CEN_2010, Hispanic_CEN_2010, 
                                                 NH_White_alone_CEN_2010, NH_Blk_alone_CEN_2010, NH_Asian_alone_CEN_2010, 
                                                 NH_AIAN_alone_CEN_2010, NH_NHOPI_alone_CEN_2010, NH_SOR_alone_CEN_2010,
                                                 pct_College_ACS_14_18, pct_Civ_unemp_16p_ACS_14_18, pct_Prs_Blw_Pov_Lev_ACS_14_18, 
                                                 Med_HHD_Inc_ACS_14_18, Med_House_Value_ACS_14_18)) %>%
  rename("tract" = "GIDTR") %>%
  mutate(PopDensity = Tot_Population_CEN_2010 / LAND_AREA,
         Race_Other =  (NH_AIAN_alone_CEN_2010 + NH_NHOPI_alone_CEN_2010 + NH_SOR_alone_CEN_2010) / Tot_Population_CEN_2010,
         Race_Hispanic = Hispanic_CEN_2010 / Tot_Population_CEN_2010,
         Race_White = NH_White_alone_CEN_2010 / Tot_Population_CEN_2010,
         Race_Black = NH_Blk_alone_CEN_2010 / Tot_Population_CEN_2010,
         Race_Asian = NH_Asian_alone_CEN_2010 / Tot_Population_CEN_2010,
         CollegeGrad = pct_College_ACS_14_18 / 100,
         Poverty = pct_Prs_Blw_Pov_Lev_ACS_14_18 / 100,
         Unemployment = pct_Civ_unemp_16p_ACS_14_18 / 100,
         MedianHHIncome = as.numeric(gsub("\\,", "", gsub("\\$", "", Med_HHD_Inc_ACS_14_18))),
         MedianHomeValue = as.numeric(gsub("\\,", "", gsub("\\$", "", Med_House_Value_ACS_14_18)))) %>%
  dplyr::select(-c(Tot_Population_CEN_2010, Hispanic_CEN_2010, NH_White_alone_CEN_2010, NH_Blk_alone_CEN_2010, NH_Asian_alone_CEN_2010, 
                   NH_AIAN_alone_CEN_2010, NH_NHOPI_alone_CEN_2010, NH_SOR_alone_CEN_2010,
                   pct_College_ACS_14_18, pct_Prs_Blw_Pov_Lev_ACS_14_18, pct_Civ_unemp_16p_ACS_14_18,
                   Med_HHD_Inc_ACS_14_18, Med_House_Value_ACS_14_18)) %>% replace(is.na(.), 0)

# 6037800204 has two rows, one with race only, one with others
DUP <- DEMOGRAPHICS %>% filter(tract == 6037800204)
DUP <- DUP %>% adorn_totals("row")
DUP = DUP[-1:-2,]
DUP[DUP$tract == 'Total']$tract = 6037800204
DEMOGRAPHICS <- DEMOGRAPHICS %>% filter(tract != 6037800204) # 8057
DEMOGRAPHICS <- rbind(DEMOGRAPHICS, DUP) #8058


## Merge ##
TRACT <- left_join(POPULATION_CENTROIDS, TRACT_HPI, by = "tract")
TRACT <- left_join(TRACT, TRACT_VOTES, by = "tract") 
TRACT <- left_join(TRACT, DEMOGRAPHICS, by = "tract")


### Comments
# Cannot find insurance type, only have % of population with one/two/no insurance at the tract level
# Not sure if the population density is computed correctly

write.csv(TRACT, "Data/TRACT_merged.csv", row.names = F)

################################################################################

### Update: insurance type merged
TRACT <- read.csv("Data/TRACT_merged.csv")
TRACT_INSURANCE = read.csv("Data/Raw/ACS/ACSDT5Y2019.B27010.csv")


HEALTH = TRACT_INSURANCE %>%
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
  dplyr::select(c(tract, Health_Employer, Health_Medicare, Health_Medicaid, Health_Other, Health_None)) %>%
  distinct()


TRACT_MERGED = left_join(TRACT, HEALTH_CA, by = "tract")


write.csv(TRACT_MERGED, "Data/TRACT_merged.csv")


################################################################################
### 2. Zip level###
# ZIP_VOTES  <- read.csv("Data/Zip_votes.csv", stringsAsFactors = FALSE)
# DEMOGRAPHICS <- read.csv("Data/Zip_demographics.csv", stringsAsFactors = FALSE) %>% rename("zip" = "Zip")
# 
# ZIP <- left_join(DEMOGRAPHICS, ZIP_VOTES, by = "zip")
# write.csv(ZIP, "Data/ZIP_merged.csv", row.names = F)





