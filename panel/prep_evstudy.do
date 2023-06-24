gl datadir "/export/storage_covidvaccine/Data"
gl logdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Logs"

cap log close
log using $logdir/prep_evstudy.log , replace

import delim $datadir/Raw/JAN_MAY.csv, clear
drop v1 x

rename (distnearestopen logdistnearestopen) (dist logdist)

destring race* hpi hpiquartile lat lng, replace force

* Calculate % of population newly vaccinated each week 
drop if population == "NA"
destring population, replace
gen vaxnewrate = vaxnew / population

* Set up panel structure
rename date date_str
gen date = date(date_str,"YMD")
format date %td
gen week = wofd(date)
format week %tw
xtset zip week

* Create lagged variable for cumulative % fully vaccinated in previous week
gen unvax_pop = population12up - vaxfull
summ unvax_pop, d
replace unvax_pop = 0 if unvax_pop < 0 
gen unvax_pct = unvax_pop/population12up

gen newvax_shareunvax = vaxnew/unvax_pop
replace newvax_shareunvax =. if newvax_shareunvax<0 | newvax_shareunvax>.3
gen asinh_newvaxshareunvax = asinh(newvax_shareunvax)

save $datadir/Analysis/panel.dta, replace

use $datadir/Analysis/panel.dta, replace

cap log close
