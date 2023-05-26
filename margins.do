global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
global codedir /mnt/staff/zhli/VaxDemandDistance
use $datadir/MAR01_vars, clear

global controlvars race_black race_asian race_hispanic race_other ///
health_employer health_medicare health_medicaid health_other collegegrad ///
unemployment poverty medianhhincome medianhomevalue popdensity population

// 1. original model
reg vaxfull c.logdistnearest##b4.hpiquartile $controlvars, robust 

loc rep "" // "q" or "a" or "" or "q1"
preserve
if "`rep'" == "q" {
// replace controls with means within hpi quartile
	foreach vv of varlist $controlvars{
		qui bysort hpiquartile: egen mean_`vv' = mean(`vv')
		qui replace `vv' = mean_`vv'
	}
}
else if "`rep'" == "a" {
// replace controls with means across all observations
	foreach vv of varlist $controlvars{
		qui egen mean_`vv' = mean(`vv')
		qui replace `vv' = mean_`vv'
	}
}
else if "`rep'" == "q1" {
	keep if hpiquartile == 1
}

margins [aweight=population], at(logdistnearest=(-0.5(0.1)2.5)) over(hpiquartile) saving(mall, replace)
combomarginsplot mall, labels("HPI Quartile 1" "HPI Quartile 2" "HPI Quartile 3" "HPI Quartile 4") ///
	ciopt(recast(rarea) color(%25)) plotopt(msize(vsmall)) ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$codedir/plots/marg1rep`rep'.png", replace
restore
//NOTE: verified that running margins on just q1 is the same as not replacing anything with means.
//also, running margins without replacement is distinct from both replacing with quartile means and with population means
// i.e., running margins without replacement is the way to go



// 2. logit


use $datadir/MAR01_vars, clear
reg delta c.logdistnearest##b4.hpiquartile $controlvars
// ereturn list



margins [aweight=population], at(logdistnearest=(-0.5(0.1)2.5)) over(hpiquartile) saving(mdelta, replace)
use mdelta, clear
gen predicted_shares = exp(_margin) / (1 + exp(_margin))




// create the plot
twoway line predicted_shares _at1 if _by1  == 1, sort || ///
       line predicted_shares _at1 if _by1  == 2, sort || ///
       line predicted_shares _at1 if _by1  == 3, sort || ///
       line predicted_shares _at1 if _by1  == 4, sort ///
       legend(order(1 "hpiquartile=1" 2 "hpiquartile=2" 3 "hpiquartile=3" 4 "hpiquartile=4"))  ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$codedir/plots/marg2.png", replace

       
// combomarginsplot mall, labels("HPI Quartile 1" "HPI Quartile 2" "HPI Quartile 3" "HPI Quartile 4") ///
// 	ciopt(recast(rarea) color(%25)) plotopt(msize(vsmall)) ///
// 	graphregion(margin(b-4 l+1)) title("") ///
// 	legend(size(small) region(color(none))) ///
// 	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
// 	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
// 	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
// 	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
// graph export "$codedir/plots/marg2.png", replace




/*

// use the logit command for easier margins
use $datadir/MAR01_vars, clear
expand 2, gen(y)
gen wgt = population * shares if y == 1
replace wgt = population * shares_out if y == 0
replace wgt = round(wgt, 1)
logit y c.logdistnearest##b4.hpiquartile $controlvars [pweight=wgt], vce(cluster zip) //TODO: p/a/f weight?

qui margins [aweight=population], at(logdistnearest=(-0.5(0.1)2.5)) over(hpiquartile) saving(mall, replace)
qui combomarginsplot mall, labels("HPI Quartile 1" "HPI Quartile 2" "HPI Quartile 3" "HPI Quartile 4") ///
	ciopt(recast(rarea) color(%25)) plotopt(msize(vsmall)) ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$codedir/plots/marg2.png", replace




// verify equivalence between logit methods (reg delta ... vs logit y ...)

clear all
global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
use $datadir/MAR01_vars, clear
reg delta logdistnearest [fweight=population]
margins [aweight=population], at(logdistnearest=(-0.5(0.1)2.5)) post //to compare
mat lis e(b)

foreach ii of numlist 1/31{
	loc preddelta = e(b)[1,`ii']
	loc predprob = exp(`preddelta')/(1+exp(`preddelta'))
	di `predprob'
}


clear all
global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
use $datadir/MAR01_vars, clear
qui expand 2, gen(y)
qui gen wgt = population * shares if y == 1
qui replace wgt = population * shares_out if y == 0
qui replace wgt = round(wgt, 1)
logit y logdistnearest [fweight=wgt], vce(cluster zip)
margins [aweight=population], at(logdistnearest=(-0.5(0.1)2.5))
*/


// binscatter2 delta logdistnearest, controls(race_black race_asian race_hispanic race_other health_employer health_medicare health_medicaid health_other collegegrad unemployment poverty medianhhincome medianhomevalue popdensity population) savegraph("/mnt/staff/zhli/binsc_delta.png") replace
// binscatter2 logshares logdistnearest, controls(race_black race_asian race_hispanic race_other health_employer health_medicare health_medicaid health_other collegegrad unemployment poverty medianhhincome medianhomevalue popdensity population) savegraph("/mnt/staff/zhli/binsc_logshares.png") replace
