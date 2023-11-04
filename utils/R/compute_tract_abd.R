library(tidyverse)
library(zipcodeR)
library(foreign)
library(jtools)
library(stargazer)

setwd("/export/storage_covidvaccine/")

###############################
# Cross-Sectional Regressions #
###############################
MAR01 = read.csv("Data/Raw/notreallyraw/MAR01.csv")
MAR01$HPIQuartile = relevel(as.factor(MAR01$HPIQuartile), ref = "4")

VaxDist = lm(VaxFull ~ LogDistNearest, data = MAR01)
VaxDistHPI = lm(VaxFull ~ LogDistNearest + HPIQuartile, data = MAR01) 
VaxDistxHPI = lm(VaxFull ~ LogDistNearest * HPIQuartile, data = MAR01)  

stargazer(VaxDist, VaxDistHPI, VaxDistxHPI, star.cutoffs = c(0.05, 0.01, 0.001), digits = 3, type = "text", no.space = TRUE, omit.stat = c("ser","f"), dep.var.labels = c("Fully Vaccinated"))

MAR01$PredictedVaxFull <- predict(VaxDist)
MAR01$ErrorTerm <- MAR01$VaxFull - MAR01$PredictedVaxFull
MAR01$ABD <- 0.75532 + MAR01$ErrorTerm # some have abd > 1 as the vaccination rate = 1, but distance far

MAR01 <- MAR01 %>% select(c(Zip, VaxFull, LogDistNearest, PredictedVaxFull, ErrorTerm, ABD))

###############################
TRACT_ZIP = read.csv("Data/Intermediate/tract_zip_crosswalk.csv")

TRACT_ABD <- left_join(TRACT_ZIP, MAR01, by = c("zip" = "Zip")) # 20026
TRACT_ABD <- na.omit(TRACT_ABD) # 19821
TRACT_ABD <- TRACT_ABD %>% group_by(tract) %>% summarize(abd = sum(frac_of_tract_area * ABD))

###############################
# TRACT = read.csv("Data/Intermediate/TRACT_merged.csv")
TRACT = read.csv("Data/Intermediate/tract_hpi_nnimpute.csv")
TRACT <- left_join(TRACT, TRACT_ABD, by = "tract")

write.csv(TRACT, "Data/Intermediate/tract_abd.csv", row.names = FALSE)









