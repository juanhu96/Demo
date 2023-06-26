cap log close
log using $logdir/evstudy.log , replace

// grab the coefplot command file
qui do "/mnt/staff/zhli/VaxDemandDistance/coefplot_cmd.do"


use $datadir/Analysis/panel.dta, replace


* first and last week of data
summ week 
global firstweek = r(min)
global lastweek = r(max)


// define treatment as getting the first site within $thr either this/last week depending on $lag01
global thr 30 //distance threshold
global lag01 = 0
global lag12 = $lag01+1
gen treated_thisweek = L${lag01}.dist < $thr & L${lag12}.dist > $thr & week > $firstweek + 1

//treatment intensity: new nearest distance upon dropping below 30km
foreach dthresh of numlist 10(10)$thr{ //10,20,...
	bys zip (week): gen lowered_to`dthresh' = L${lag01}.dist < `dthresh' & L${lag01}.dist > `dthresh'-10 & L${lag12}.dist > $thr & week > $firstweek + 1
	bys zip: egen treated`dthresh' = max(lowered_to`dthresh')
}

bys zip: egen treated = max(treated_thisweek)
tab treated

// treatment period
bys zip: egen treatpd = min(treated_thisweek * week) if treated_thisweek
bys zip (treatpd): replace treatpd = treatpd[1] 
format treatpd %tw

replace treatpd = 0 if missing(treatpd)

//for simple TWFE
gen after = treated & week>treatpd
foreach dthresh of numlist 10(10)$thr{
	gen after_`dthresh' = treated`dthresh' & week>treatpd
}

//for event study
gen eventtime = week - treatpd if treated
tab eventtime if !missing(eventtime), gen(ieventtime)
drop ieventtime12 //base level
foreach vv of varlist ieventtime*{
	replace `vv' = 0 if !treated
}



//treatment time tabulation
tabstat treated_thisweek, by(week) stat(count sum mean)


// save data for event study regression
save $datadir/Analysis/Demand/panel_toreg.dta, replace
use $datadir/Analysis/Demand/panel_toreg.dta, clear


// global yvar newvax_shareunvax
global yvar partfull


//can also use the numerator
//create doses per person


/*
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
graph save $outdir/`yvar'/trendline/wk`wkinyear'.pdf, replace
restore
}
}
*/



/*
original: reghdfe yvar logdist control a(zip week)
new: reghdfe yvar change_logdist control a(zip week)
0 0 0 0 0 change_logdist change_logdist change_logdist change_logdist change_logdist

cap initial at 100km
define treatment at the first time something opens within 30km
after the next time the nearest dist changes, can delete the observations , or keep 
*/

// generate treatment intensity variable: change in log(dist) upon dropping below 30km
global dist_cap = 100
gen dist_forchange = dist
replace dist_forchange = $dist_cap if dist > $dist_cap
replace dist_forchange = log(dist_forchange)
bys zip (week): gen change_logdist = dist_forchange - L.dist_forchange //change in log distance
replace change_logdist = 0 if !treated_thisweek | missing(change_logdist) | change_logdist>0
replace change_logdist = -change_logdist

bys zip: egen treatment_scaling = max(change_logdist) 
assert treatment_scaling ==0 if !treated
foreach vv of varlist after ieventtime*{
	gen scaled_`vv' = `vv' * treatment_scaling
}

// br zip week *dist* *treat* *sc* *after* if treated
//TWFE

//simple ATT
reghdfe $yvar after [aweight=population12up], absorb(zip week#hpiquartile) vce(cluster zip)

//treatment intensity by distance 
reghdfe $yvar after_* [aweight=population12up], absorb(zip week#hpiquartile) vce(cluster zip)

//event study
reghdfe $yvar ieventtime* [aweight=population12up], absorb(zip week#hpiquartile) vce(cluster zip)
coefplot_cmd, regtype("twfe") outfile("$outdir/$yvar/coefplot/twfe_thr${thr}.pdf") note("Using ${thr}km as the threshold for treatment.")

//simple ATT, scaled by change in distance upon dropping below 30km
reghdfe $yvar scaled_after [aweight=population12up], absorb(zip week#hpiquartile) vce(cluster zip)

//event study scaled by change in distance upon dropping below 30km
reghdfe $yvar scaled_ieventtime* [aweight=population12up], absorb(zip week#hpiquartile) vce(cluster zip)
coefplot_cmd, regtype("twfe") outfile("$outdir/$yvar/coefplot/twfe_thr${thr}_scaled.pdf") note("Using ${thr}km as the threshold. Scaling treatment by change in distance.")


//CSDID

// simple ATT
csdid $yvar [weight=population12up], ivar(zip) time(week) gvar(treatpd) agg(simple)


foreach yvar in partfull{ //newvax_shareunvax
	csdid `yvar' [weight=population12up], ivar(zip) time(week) gvar(treatpd) agg(event) 
	coefplot_cmd, regtype("csdid") outfile("$outdir/`yvar'/coefplot/csdid_thr${thr}.pdf")
}


//by intensity
foreach intensity in 10 20 30{
	loc intensity_lb = `intensity' - 10
	csdid $yvar if !treated | treated`intensity' [weight=population12up], ivar(zip) time(week) gvar(treatpd) agg(event) 
	coefplot_cmd, regtype("csdid") outfile("$outdir/$yvar/coefplot/csdid_onlytreat`intensity'_thr${thr}.pdf") note("Restricting treatment group to `intensity_lb'-`intensity'km after treatment")
}

//redefining treatment 
cap drop treated* 
foreach thr in 10 20 30{
	bys zip (week): gen treated_thisweek`thr' = dist < $thr & L.dist > $thr & week > $firstweek + 1
	bys zip: egen treated`thr' = max(treated_thisweek`thr')

	// treatment period
	bys zip: egen treatpd`thr' = min(treated_thisweek`thr' * week) if treated_thisweek`thr'
	bys zip (treatpd`thr'): replace treatpd`thr' = treatpd`thr'[1] 
	format treatpd`thr' %tw

	replace treatpd`thr' = 0 if missing(treatpd`thr')

	csdid $yvar [weight=population12up], ivar(zip) time(week) gvar(treatpd`thr') agg(event) 
	coefplot_cmd, regtype("csdid") outfile("$outdir/$yvar/coefplot/csdid_thr`thr'.pdf") note("Using `thr'km as the threshold for treatment")
}








cap log close



