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
setwd("/mnt/phd/jihu/COVID")
################################################################################
### 1. Tract level ###

## Import ##
TRACT_VOTES <- read.csv("Data/Tract_votes.csv", stringsAsFactors = FALSE)
TRACT_HPI <- read.csv("Data/HPItract2022.csv", stringsAsFactors = FALSE)
POPULATION_CENTROIDS <- read.csv("Data/Tract_centroids.csv", stringsAsFactors = FALSE)
DEMOGRAPHICS <- read.csv("Data/Tract_demographics.csv", stringsAsFactors = FALSE)

## Rename ##
TRACT_HPI <- TRACT_HPI %>% rename("tract" = "geoid")
POPULATION_CENTROIDS <- POPULATION_CENTROIDS %>% rename("tract" = "GEOID") %>% select(-c(ID))
# Use Census 2010 since it also matches the population in POPULATION_CENTROIDS
DEMOGRAPHICS <- DEMOGRAPHICS %>% select(c(GIDTR, LAND_AREA,
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
  select(-c(Tot_Population_CEN_2010, Hispanic_CEN_2010, NH_White_alone_CEN_2010, NH_Blk_alone_CEN_2010, NH_Asian_alone_CEN_2010, 
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
### 2. Zip level###
ZIP_VOTES  <- read.csv("Data/Zip_votes.csv", stringsAsFactors = FALSE)
DEMOGRAPHICS <- read.csv("Data/Zip_demographics.csv", stringsAsFactors = FALSE) %>% rename("zip" = "Zip")

ZIP <- left_join(DEMOGRAPHICS, ZIP_VOTES, by = "zip")
write.csv(ZIP, "Data/ZIP_merged.csv", row.names = F)




