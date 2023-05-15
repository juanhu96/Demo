global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
global outdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Output"
global logdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Logs"
global coefplotdofile "/mnt/staff/zhli/VaxDemandDistance/coefplot_cmd.do"
cap log close
log using $logdir/evstudy.log , replace

use $datadir/Analysis/panel.dta, replace


* first and last week of data
summ week 
global firstweek = r(min)
global lastweek = r(max)


// define treatment as getting the first site within $tr_dist either this/last week depending on $lag01
global tr_dist 30 //distance threshold
global lag01 = 0
global lag12 = $lag01+1
gen treated_thisweek = L${lag01}.dist < $tr_dist & L${lag12}.dist > $tr_dist & week > $firstweek + 1

//treatment intensity: new nearest distance upon dropping below 30km
foreach dthresh of numlist 10(10)$tr_dist{ //10,20,...
	bys zip (week): gen lowered_to`dthresh' = L${lag01}.dist < `dthresh' & L${lag01}.dist > `dthresh'-10 & L${lag12}.dist > $tr_dist & week > $firstweek + 1
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


//treatment time tabulation
tabstat treated_thisweek, by(week) stat(count sum mean)



global yvar newvax_shareunvax
global yvar partfull

//can also use the numerator
//create doses per person




//line graphs
foreach yvar in partfull full{
foreach wkinyear in 4 5 6 7 8 9{
preserve
// keep if treatpd==$firstweek+2
loc wk = `wkinyear' - 2
loc xlinewk = `wk'+$firstweek
keep if inlist(treatpd, 0, `xlinewk')
count if treated_thisweek
loc ntreated = r(N)
qui gcollapse (mean) y = `yvar', by(week treated)
// twoway line y week
qui reshape wide y, j(treated) i(week)
twoway line y0 week || ///
line y1 week, ///
legend(order(1 "Never treated" 2 "Treated on week `wkinyear' (`ntreated' ZIP codes)")) ///
xline(`xlinewk') ///
title("Mean of `yvar'") 
graph display, margin(r+5)
graph save $outdir/mean_`yvar'_wk`wkinyear'.pdf, replace
restore
}
}



//TWFE
/*

//simple ATT
reghdfe $yvar after [aweight=population12up], absorb(zip week) vce(cluster zip)

//treatment intensity by distance 
reghdfe $yvar after_* [aweight=population12up], absorb(zip week) vce(cluster zip)

//event study
reghdfe $yvar ieventtime* [aweight=population12up], absorb(zip week#hpiquartile) vce(cluster zip)


//make plot
mat coef =  e(b)[1,"ieventtime1".."ieventtime26"]
mat varmat = e(V)["ieventtime1".."ieventtime26", "ieventtime1".."ieventtime26"]
local xlabels 1 "-12" 7 "-6" 13 "0" 19 "6" 25 "12"
coefplot (matrix(coef), lwidth(thick) recast(line) v(varmat) ciopt(recast(rarea) color(navy%15)) axis(1)), ///
	vertical nooffsets legend(off) ///
	yline(0, lcolor(gray%50)) ///
	xline(13, lcolor(gray) lpattern(dash)) ///
	xlab($xlabels) xtitle("Weeks Since Treatment") title("TWFE")

graph display
graph export  "$outdir/vaxcoefplot_twfe.pdf", replace
*/

//CSDID

// simple ATT
csdid $yvar [weight=population12up], ivar(zip) time(week) gvar(treatpd) agg(simple)


foreach yvar in newvax_shareunvax partfull{
	global yvar `yvar'
	// event study
	csdid $yvar [weight=population12up], ivar(zip) time(week) gvar(treatpd) agg(event) 
	qui do $coefplotdofile
	graph export  "$outdir/coef_csdid_${yvar}.pdf", replace


	csdid $yvar if !treated | treated10 [weight=population12up], ivar(zip) time(week) gvar(treatpd) agg(event) 
	qui do $coefplotdofile
	graph export  "$outdir/coef_csdid_${yvar}_onlytreat10.pdf", replace
}

/*
1. less than 30
2. less than 10 (exclude 10-30)
3. some evidence that the pre-trends are not good (the line graphs)
4. matched based on pre-opening vax rate (for every treated zip, match to 3-5 that had the closest vax rate in the week before treatment) and hpiquartile (exact match). Maybe also pre-treatment distance. See PE paper - cohort#time and facility#cohort FEs. 
*/



cap log close



