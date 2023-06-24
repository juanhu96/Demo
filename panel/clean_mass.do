// Snippet to clean the California Mass Vaccination Site Data
gl datadir "/export/storage_covidvaccine/Data"

import delim $datadir/Raw/CaliforniaMassVaccination.csv, clear

gen opendate = date(opened,"MDY")
gen closedate = date(closed,"MDY")
format opendate closedate %td
drop opened closed 
rename (opendate closedate) (opened closed)
cap drop week 
gen week = wofd(opened)
format week %tw 
save $datadir/Intermediate/CaliforniaMassVaccination.dta, replace


use $datadir/Intermediate/CaliforniaMassVaccination.dta, clear

//look at closing dates/ dates where distance of nearest site increased
