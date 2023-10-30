################################################################################

# Input (all in raw form): 
# tract-level population centroids, tract-level HPI, locations of pharmacy/dollar stores/...

# Output: 
# tract-level population centroids with HPI info, vector of demand/HPI quartiles, 
# distance matrices for tract & pharmacy/dollar stores/..., cleaned up locations of pharmacy/dollar stores/...

library(Matrix)
library(dplyr)
library(janitor)

library(foreign)
library(shapefiles)

# Comment: I believe some of the libraries are redundant...
library(proxy)
library(sp)
library(sf)
library(rgdal)
library(MASS)

library(parallel)
library(raster)
library(foreach)
library(doParallel)
library(geosphere)
library(doRNG)

setwd("/export/storage_covidvaccine")


################################################################################
################################################################################
################################################################################


### Population centroids ### 
POPULATION_CENTROIDS <- as.data.frame(st_read("Data/Raw/Location/popctr_tracts2010/popctr_tracts2010.shp"))
POPULATION_CENTROIDS <- POPULATION_CENTROIDS %>% 
  dplyr::select(-c('geometry')) %>%
  dplyr::rename(GEOID = FIPS, STATE_ID = STATE, COUNTY_ID = COUNTY, TRACT_ID = TRACT)

POPULATION_CENTROIDS <- POPULATION_CENTROIDS %>% dplyr::filter(STATE_ID == '06') # CA
POPULATION_CENTROIDS$ID = seq.int(nrow(POPULATION_CENTROIDS))

write.table(POPULATION_CENTROIDS$POPULATION, file=paste("Data/CA_demand.csv", sep = ""), row.names = F, col.names = F) # model input: vector of demand
write.table(POPULATION_CENTROIDS$GEOID, file="Data/CA_tractID.csv", row.names = F, col.names = F)
write.csv(POPULATION_CENTROIDS, file=paste("Data/tract_centroids.csv", sep = ""), row.names = F)


################################################################################
################################################################################
################################################################################


### Obtain demand/population for 5+ only
DEMOGRAPHIC <- read.csv("Data/Raw/Census/pdb2020trv2_us.csv")
## Tot_Population_CEN_2010 matches the population from population centroid
DEMOGRAPHIC <- DEMOGRAPHIC %>% filter(State == 6) %>% dplyr::select(c(GIDTR, State, State_name, County, County_name, Tract,
                                                                      Tot_Population_CEN_2010, Tot_Population_ACS_14_18, Tot_Population_ACSMOE_14_18,
                                                                      Pop_under_5_CEN_2010, Pop_under_5_ACS_14_18, Pop_under_5_ACSMOE_14_18)) %>%
  mutate(Pop_over_5_CEN_2010 = Tot_Population_CEN_2010 - Pop_under_5_CEN_2010)

DUP <- DEMOGRAPHIC %>% filter(GIDTR == 6037800204)
DUP <- DUP %>% adorn_totals("row")
DUP = DUP[-1:-2,]
DUP <- DUP %>% mutate(GIDTR = 6037800204, State = 6, State_name = 'California', County = 37, County_name = 'Los Angeles County', Tract = 800204)
DEMOGRAPHIC <- DEMOGRAPHIC %>% filter(GIDTR != 6037800204) # 8057
DEMOGRAPHIC <- rbind(DEMOGRAPHIC, DUP) #8058

POPULATION_CENTROIDS <- read.csv("Data/tract_centroids.csv")
POPULATION_CENTROIDS <- left_join(POPULATION_CENTROIDS, DEMOGRAPHIC, by = c('GEOID' = 'GIDTR'))

write.table(POPULATION_CENTROIDS$Pop_over_5_CEN_2010, file=paste("Data/CA_demand_over_5.csv", sep = ""), row.names = F, col.names = F) # model input: vector of demand


################################################################################
################################################################################
################################################################################


### HPI for each tract ###
TRACT_HPI <- read.csv("Data/Raw/HPItract2022.csv")
TRACT_HPI$GEOID <- paste("0", TRACT_HPI$geoid, sep="")
TRACT_HPI <- TRACT_HPI %>% dplyr::select(-c('geoid'))
POPULATION_CENTROIDS <- left_join(POPULATION_CENTROIDS, TRACT_HPI, by = "GEOID")
POPULATION_CENTROIDS[is.na(POPULATION_CENTROIDS)] <- 0

write.table(POPULATION_CENTROIDS$HPIQuartile, file=paste("Data/HPIQuartile_TRACT.csv", sep = ""), row.names = F, col.names = F) # model input: vector of HPI quartile


################################################################################
################################################################################
################################################################################


### DISTANCE MATRIX  ### 

### Set up parallel computing ###
numCores <- detectCores()
cl <- makeCluster(20) # number of cores to use on the server, e.g. numCores/2 
registerDoParallel(cl)

### Current pharmacies ### 
CURRENT <- read.csv("Data/Raw/Location/00_Pharmacies.csv", stringsAsFactors = FALSE)
CURRENT_STATE = CURRENT[CURRENT$state %in% c('CA'), ]
CURRENT_STATE$latitude = as.numeric(CURRENT_STATE$latitude)
CURRENT_STATE$longitude = as.numeric(CURRENT_STATE$longitude)
CURRENT_STATE$ID = seq.int(nrow(CURRENT_STATE))
rm(CURRENT)

dist_values_current = foreach(store_id = 1:nrow(CURRENT_STATE), .packages="foreach") %dopar% {
  store_row = CURRENT_STATE[CURRENT_STATE["ID"] == store_id, ]
  store_lat = store_row$latitude
  store_lon = store_row$longitude
  foreach(tract_id = 1:nrow(POPULATION_CENTROIDS), .packages="foreach", .combine=c) %dopar%{
    tract_row = POPULATION_CENTROIDS[POPULATION_CENTROIDS["ID"] == tract_id,]
    tract_lat = tract_row$LATITUDE
    tract_lon = tract_row$LONGITUDE
    dist = round(geosphere::distHaversine(c(store_lon, store_lat), c(tract_lon, tract_lat)))
  }
}

dist_matrix_current <- matrix(0, nrow = nrow(CURRENT_STATE), ncol = nrow(POPULATION_CENTROIDS))
for(store_id in 1:nrow(CURRENT_STATE)){
  for(j in 1:nrow(POPULATION_CENTROIDS)){
    tract_id = j
    dist_matrix_current[store_id, tract_id] <- dist_values_current[[store_id]][j]
  }
}

dist_matrix_current = as.matrix(dist_matrix_current)
write.table(dist_matrix_current, file = paste("Data/CA_dist_matrix_current.csv", sep=""), sep=',',
            row.names = FALSE, col.names = FALSE)

rm(dist_values_current)
rm(dist_matrix_current)
stopCluster(cl)

################################################################################
### Dollar stores ### 
numCores <- detectCores()
cl <- makeCluster(20)
registerDoParallel(cl)

DOLLAR <- read.csv("Data/Raw/Location/01_DollarStores.csv", stringsAsFactors = FALSE)
DOLLAR_STATE = DOLLAR[DOLLAR$State %in% c('CA'), ]
DOLLAR_STATE$ID = seq.int(nrow(DOLLAR_STATE))
rm(DOLLAR)

dist_values = foreach(store_id = 1:nrow(DOLLAR_STATE), .packages="foreach") %dopar% {
  store_row = DOLLAR_STATE[DOLLAR_STATE["ID"] == store_id, ]
  store_lat = store_row$Latitude
  store_lon = store_row$Longitude
  
  foreach(tract_id = 1:nrow(POPULATION_CENTROIDS), .packages="foreach", .combine=c) %dopar%{
    tract_row = POPULATION_CENTROIDS[POPULATION_CENTROIDS["ID"] == tract_id,]
    tract_lat = tract_row$LATITUDE
    tract_lon = tract_row$LONGITUDE
    dist = round(geosphere::distHaversine(c(store_lon, store_lat), c(tract_lon, tract_lat)))
  }
}

dist_matrix <- matrix(0, nrow = nrow(DOLLAR_STATE), ncol = nrow(POPULATION_CENTROIDS))
for(store_id in 1:nrow(DOLLAR_STATE)){
  for(j in 1:nrow(POPULATION_CENTROIDS)){
    tract_id = j
    dist_matrix[store_id, tract_id] <- dist_values[[store_id]][j]
  }
}

dist_matrix = as.matrix(dist_matrix)
write.table(dist_matrix, file= paste("Data/CA_dist_matrix_Dollar.csv", sep=""), sep=',',
            row.names = FALSE, col.names = FALSE)
rm(dist_values)
rm(dist_matrix)
stopCluster(cl)

################################################################################
### Others (except 03_FastFood) ###
store_type = c("02_DiscountRetailers", "04_Coffee",
               "05_ConvenienceStores", "06_GasStations", "07_CarDealers",
               "08_PostOffices", "09_PublicSchools", "10_Libraries")

for(type in store_type){
  
  numCores <- detectCores()
  cl <- makeCluster(numCores/2)
  registerDoParallel(cl)
  
  STORE <- read.csv(paste("Data/Raw/Location/", type, ".csv", sep = ""), stringsAsFactors = FALSE)
  STORE$ID = seq.int(nrow(STORE))
  
  dist_values = foreach(store_id = 1:nrow(STORE), .packages="foreach") %dopar% {
    store_row = STORE[STORE["ID"] == store_id, ]
    store_lat = store_row$latitude
    store_lon = store_row$longitude
    
    foreach(tract_id = 1:nrow(POPULATION_CENTROIDS), .packages="foreach", .combine=c) %dopar%{
      tract_row = POPULATION_CENTROIDS[POPULATION_CENTROIDS["ID"] == tract_id,]
      tract_lat = tract_row$LATITUDE
      tract_lon = tract_row$LONGITUDE
      dist = round(geosphere::distHaversine(c(store_lon, store_lat), c(tract_lon, tract_lat)))
    }
  }
  
  dist_matrix <- matrix(0, nrow = nrow(STORE), ncol = nrow(POPULATION_CENTROIDS))
  for(store_id in 1:nrow(STORE)){
    for(j in 1:nrow(POPULATION_CENTROIDS)){
      tract_id = j
      dist_matrix[store_id, tract_id] <- dist_values[[store_id]][j]
    }
  }
  
  dist_matrix = as.matrix(dist_matrix)
  write.table(dist_matrix, file= paste("Data/CA_dist_matrix_", substring(type,4), ".csv", sep=""), sep=',',
              row.names = FALSE, col.names = FALSE)
  rm(dist_values)
  rm(dist_matrix)
  stopCluster(cl)
}

################################################################################
### Fast food: Mcdonald only ###
numCores <- detectCores()
cl <- makeCluster(numCores/2)
registerDoParallel(cl)

STORE <- read.csv(paste("Data/Raw/Location/03_FastFood.csv", sep = ""), stringsAsFactors = FALSE)
STORE <- STORE %>% filter(location_name == "McDonald's")
STORE$ID = seq.int(nrow(STORE))

dist_values = foreach(store_id = 1:nrow(STORE), .packages="foreach") %dopar% {
  store_row = STORE[STORE["ID"] == store_id, ]
  store_lat = store_row$latitude
  store_lon = store_row$longitude
  
  foreach(tract_id = 1:nrow(POPULATION_CENTROIDS), .packages="foreach", .combine=c) %dopar%{
    tract_row = POPULATION_CENTROIDS[POPULATION_CENTROIDS["ID"] == tract_id,]
    tract_lat = tract_row$LATITUDE
    tract_lon = tract_row$LONGITUDE
    dist = round(geosphere::distHaversine(c(store_lon, store_lat), c(tract_lon, tract_lat)))
  }
}

dist_matrix <- matrix(0, nrow = nrow(STORE), ncol = nrow(POPULATION_CENTROIDS))
for(store_id in 1:nrow(STORE)){
  for(j in 1:nrow(POPULATION_CENTROIDS)){
    tract_id = j
    dist_matrix[store_id, tract_id] <- dist_values[[store_id]][j]
  }
}

dist_matrix = as.matrix(dist_matrix)
write.table(dist_matrix, file= paste("Data/CA_dist_matrix_Mcdonald.csv", sep=""), sep=',',
            row.names = FALSE, col.names = FALSE)
rm(dist_values)
rm(dist_matrix)
stopCluster(cl)

################################################################################
################################################################################
################################################################################
# ### Locations of each chain
# store_type = c("02_DiscountRetailers", "04_Coffee",
#                "05_ConvenienceStores", "06_GasStations", "07_CarDealers",
#                "08_PostOffices", "10_Libraries")
# 
# for(type in store_type){
#   STORE <- read.csv(paste("Data/Location Data Extended/", type, ".csv", sep = ""), stringsAsFactors = FALSE)
#   STORE <- STORE %>% dplyr::select(c(location_name, brands, latitude, longitude, street_address, city,
#                               region, postal_code))
#   write.csv(STORE, file=paste("Data/Chain/", substring(type, 4), ".csv", sep = ""), row.names = F)
# }
# 
# 
# ## Fast food: Mcdonald only
# STORE <- read.csv(paste("Data/Location Data Extended/03_FastFood.csv", sep = ""), stringsAsFactors = FALSE)
# STORE <- STORE %>% filter(location_name == "McDonald's")
# STORE <- STORE %>% dplyr::select(c(location_name, brands, latitude, longitude, street_address, city,
#                                    region, postal_code))
# write.csv(STORE, file=paste("Data/Chain/Mcdonald.csv", sep = ""), row.names = F)
# 
# ## 09_HighSchools
# STORE <- read.csv(paste("Data/Location Data Extended/09_HighSchools.csv", sep = ""), stringsAsFactors = FALSE)
# STORE <- STORE %>% dplyr::select(c(County, District, School, Street, City, Zip, State,
#                                    latitude, longitude))
# write.csv(STORE, file=paste("Data/Chain/HighSchools.csv", sep = ""), row.names = F)
# 
# ## Current pharmacies
# STORE <- CURRENT_STATE
# STORE <- STORE %>% dplyr::select(c(location, address, city, state, zip_code, combined, StateID,
#                                    latitude, longitude))
# write.csv(STORE, file=paste("Data/Chain/Pharmacies.csv", sep = ""), row.names = F)



