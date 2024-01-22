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

VaxDistAbove0.5 = lm(VaxFull ~ pmax(LogDistNearest, log(0.5)), data = MAR01)
VaxDistAbove1 = lm(VaxFull ~ pmax(LogDistNearest, log(1)), data = MAR01)
VaxDistAbove1.6 = lm(VaxFull ~ pmax(LogDistNearest, log(1.6)), data = MAR01)

stargazer(VaxDist, VaxDistHPI, VaxDistxHPI, star.cutoffs = c(0.05, 0.01, 0.001), digits = 3, type = "text", no.space = TRUE, omit.stat = c("ser","f"), dep.var.labels = c("Fully Vaccinated"))
stargazer(VaxDist, VaxDistAbove0.5, VaxDistAbove1, VaxDistAbove1.6, star.cutoffs = c(0.05, 0.01, 0.001), digits = 3, type = "text", no.space = TRUE, omit.stat = c("ser","f"), dep.var.labels = c("Fully Vaccinated"))

# for prediction using the same variable, i.e. pmax(LogDistNearest, 0)
MAR01$PredictedVaxFull <- predict(VaxDist)
MAR01$PredictedVaxFull0.5 <- predict(VaxDistAbove0.5)
MAR01$PredictedVaxFull1 <- predict(VaxDistAbove1)
MAR01$PredictedVaxFull1.6 <- predict(VaxDistAbove1.6) 

MAR01$ErrorTerm <- MAR01$VaxFull - MAR01$PredictedVaxFull
MAR01$ErrorTerm0.5 <- MAR01$VaxFull - MAR01$PredictedVaxFull0.5
MAR01$ErrorTerm1 <- MAR01$VaxFull - MAR01$PredictedVaxFull1
MAR01$ErrorTerm1.6 <- MAR01$VaxFull - MAR01$PredictedVaxFull1.6

MAR01$ABD <- 0.755 + MAR01$ErrorTerm
MAR01$ABD0.5 <- 0.768 + MAR01$ErrorTerm0.5
MAR01$ABD1 <- 0.788 + MAR01$ErrorTerm1
MAR01$ABD1.6 <- 0.818 + MAR01$ErrorTerm1.6

MAR01 <- MAR01 %>% select(c(Zip, VaxFull, LogDistNearest, PredictedVaxFull, ErrorTerm, ABD, ABD0.5, ABD1, ABD1.6))

###############################
TRACT_ZIP = read.csv("Data/Intermediate/tract_zip_crosswalk.csv")

TRACT_ABD <- left_join(TRACT_ZIP, MAR01, by = c("zip" = "Zip")) # 20026
TRACT_ABD <- na.omit(TRACT_ABD) # 19821
TRACT_ABD <- TRACT_ABD %>% group_by(tract) %>% summarize(abd = sum(frac_of_tract_area * ABD),
                                                         abd0.5 = sum(frac_of_tract_area * ABD0.5),
                                                         abd1 = sum(frac_of_tract_area * ABD1),
                                                         abd1.6 = sum(frac_of_tract_area * ABD1.6))

###############################
# TRACT = read.csv("Data/Intermediate/TRACT_merged.csv")
TRACT = read.csv("Data/Intermediate/tract_hpi_nnimpute.csv")
TRACT <- left_join(TRACT, TRACT_ABD, by = "tract")

write.csv(TRACT, "Data/Intermediate/tract_abd.csv", row.names = FALSE)








