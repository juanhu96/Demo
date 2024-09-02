# install.packages("zipcodeR")
library(zipcodeR)

setwd("/export/storage_covidvaccine/Data")

zips_ca <- zip_code_db[zip_code_db$state == 'CA',c('zipcode', 'lat', 'lng')]
names(zips_ca) <- c('zip', 'latitude', 'longitude')

# save as csv
write.csv(zips_ca, file = "Intermediate/zip_coords.csv", row.names = FALSE)
