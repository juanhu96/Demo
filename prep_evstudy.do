gl datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
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

* first and last week of data
summ week 
gl firstweek = r(min)
gl lastweek = r(max)

// define treatment as getting the first site within $tr_dist either this/last week depending on $lag01
gl tr_dist 30 //distance threshold
gl lag01 = 0 
gl lag12 = $lag01+1
gen treated_thisweek = L${lag01}.dist < $tr_dist & L${lag12}.dist > $tr_dist & week > $firstweek + 1

tab week if treated_thisweek


//treatment intensity: new nearest distance upon dropping below 30km
foreach dthresh of numlist 10(10)$tr_dist{ //10,20,...
	bys zip (week): gen lowered_to`dthresh' = L.dist < `dthresh' & L.dist > `dthresh'-10 & L2.dist > $tr_dist & week > $firstweek + 1
	bys zip: egen treated`dthresh' = max(lowered_to`dthresh')
}


bys zip: egen treated = max(treated_thisweek)
tab treated

// treatment period
bys zip: egen treatpd = min(treated_thisweek * week) if treated_thisweek
bys zip (treatpd): replace treatpd = treatpd[1] 
format treatpd %tw

replace treatpd = 0 if missing(treatpd)
gen eventtime = week - treatpd if treated
tab treatpd
tab eventtime if !missing(eventtime), gen(ieventtime)
drop ieventtime12 //base level

//for TWFE
gen after = treated & week>treatpd
foreach dthresh of numlist 10(10)$tr_dist{
	gen after_`dthresh' = treated`dthresh' & week>treatpd
}

save $datadir/Analysis/panel.dta, replace

use $datadir/Analysis/panel.dta, replace

//line graph 
preserve
gcollapse (mean) y = newvax_shareunvax, by(week treated)
reshape wide y, j(treated) i(week)
twoway line y0 week || ///
line y1 week, ///
legend(order(1 "Untreated" 2 "Treated"))
restore


cap log close
