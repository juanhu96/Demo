// margins plots for quartile-based analysis

global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
global outdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Output"

foreach spec in hpi dshare income race{
	loc byvar "`spec'_quartile"

	use $datadir/Analysis/tracts_marg_by`spec', clear

	gen agent_id = _n

	expand 31, gen(expd)
	bys agent_id (expd): gen logdist_m = -0.5 if _n==1
	bys agent_id (expd): replace logdist_m = logdist_m[_n-1]+0.1 if _n!=1
	gen u_i = meanutil + distbeta*logdist_m
	gen share_i = exp(u_i)/(1+exp(u_i))

	gcollapse (mean) share_i [weight=weights], by(`byvar' logdist_m)
	twoway line share_i logdist_m if `byvar' == 1, sort || ///
		line share_i logdist_m if `byvar' == 2, sort || ///
		line share_i logdist_m if `byvar' == 3, sort || ///
		line share_i logdist_m if `byvar' == 4, sort ///
		legend(order(1 "`byvar'=1" 2 "`byvar'=2" 3 "`byvar'=3" 4 "`byvar'=4"))  ///
		graphregion(margin(b-4 l+1)) title("") ///
		legend(size(small) region(color(none))) ///
		subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
		ylabel(.4 "40%" .5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
		xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
		xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
	graph export "$outdir/margins/marg4_by`spec'.png", replace

}
