// margins plots for original model, logit, and RC

global datadir "/export/storage_covidvaccine/Data"
global outdir "/export/storage_covidvaccine/Result/Demand"

use $datadir/Analysis/Demand/MAR01_vars, clear

global controlvars race_black race_asian race_hispanic race_other ///
health_employer health_medicare health_medicaid health_other collegegrad ///
unemployment poverty medianhhincome medianhomevalue popdensity population

// 1. original model
reg vaxfull c.logdistnearest##b4.hpiquartile $controlvars, robust 

margins [aweight=population], at(logdistnearest=(-0.5(0.1)2.5)) over(hpiquartile) saving(mall, replace)
combomarginsplot mall, labels("HPI Quartile 1" "HPI Quartile 2" "HPI Quartile 3" "HPI Quartile 4") ///
	ciopt(recast(rarea) color(%25)) plotopt(msize(vsmall)) ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$outdir/margins/orig.png", replace



// 2. logit
use $datadir/Analysis//Demand/MAR01_vars, clear
reg delta c.logdistnearest##b4.hpiquartile $controlvars
predict xi, residuals
expand 31, gen(expd)
drop logdistnearest
bys zip (expd): gen logdistnearest = -0.5 if _n==1
bys zip (expd): replace logdistnearest = logdistnearest[_n-1]+0.1 if _n!=1

predict pred_delta
replace pred_delta = pred_delta + xi

gen pred_share = exp(pred_delta)/(1+exp(pred_delta))

gcollapse (mean) pred_share [weight=population], by(hpiquartile logdistnearest)

twoway line pred_share logdistnearest if hpiquartile  == 1, sort || ///
       line pred_share logdistnearest if hpiquartile  == 2, sort || ///
       line pred_share logdistnearest if hpiquartile  == 3, sort || ///
       line pred_share logdistnearest if hpiquartile  == 4, sort ///
       legend(order(1 "hpiquartile=1" 2 "hpiquartile=2" 3 "hpiquartile=3" 4 "hpiquartile=4"))  ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$outdir/marg2.png", replace


// 3. RC
use $datadir/Analysis/Demand/agents_marg, clear
gen agent_id = _n

expand 31, gen(expd)
bys agent_id (expd): gen logdist_m = -0.5 if _n==1
bys agent_id (expd): replace logdist_m = logdist_m[_n-1]+0.1 if _n!=1
gen u_i = meanutil + (distbeta+rc)*logdist_m
gen share_i = exp(u_i)/(1+exp(u_i))

gcollapse (mean) share_i [weight=population], by(hpiquartile logdist_m)
twoway line share_i logdist_m if hpiquartile  == 1, sort || ///
       line share_i logdist_m if hpiquartile  == 2, sort || ///
       line share_i logdist_m if hpiquartile  == 3, sort || ///
       line share_i logdist_m if hpiquartile  == 4, sort ///
       legend(order(1 "hpiquartile=1" 2 "hpiquartile=2" 3 "hpiquartile=3" 4 "hpiquartile=4"))  ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$outdir/marg3.png", replace


// 5. tract distances with RC (on log distance and on the constant)
use $datadir/Analysis/Demand/tracts_marg_rc, clear

gen agent_id = _n

expand 31, gen(expd)
bys agent_id (expd): gen logdist_m = -0.5 if _n==1
bys agent_id (expd): replace logdist_m = logdist_m[_n-1]+0.1 if _n!=1
gen u_i = meanutil + (logdist_m*nu_dist*coef_nuXlogdist) + distbeta*logdist_m
gen share_i =exp(u_i)/(1+exp(u_i))

gcollapse (mean) share_i [weight=weights], by(hpiquartile logdist_m)
twoway line share_i logdist_m if hpiquartile  == 1, sort || ///
       line share_i logdist_m if hpiquartile  == 2, sort || ///
       line share_i logdist_m if hpiquartile  == 3, sort || ///
       line share_i logdist_m if hpiquartile  == 4, sort ///
       legend(order(1 "hpiquartile=1" 2 "hpiquartile=2" 3 "hpiquartile=3" 4 "hpiquartile=4"))  ///
	graphregion(margin(b-4 l+1)) title("") ///
	legend(size(small) region(color(none))) ///
	subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
	ylabel(.5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
	xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
	xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
graph export "$outdir/marg5.png", replace

