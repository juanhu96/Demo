global datadir "/export/storage_covidvaccine/Data"
global outdir "/export/storage_covidvaccine/Result/Demand"

// run after demest_tracts_margins.py

cap mkdir "$outdir/margins"
cap mkdir "$outdir/margins/pooled"
cap mkdir "$outdir/margins/byqrtl"

foreach config in "000" "100"{ //pooled
    loc incl_hpiq = substr("`config'", 1, 1)
    loc incl_ctrl = substr("`config'", 3, 1)
    if `incl_hpiq'==1{
    	loc notes "Notes: Including HPI quartile controls."
    }
    use "$datadir/Analysis/Demand/marg_`config'.dta", clear
    twoway rarea share_lb share_ub logdist_m, color(gs14%50) || ///
    line share_i logdist_m , ///
        graphregion(margin(b-4 l+1)) title("") ///
        legend(off) ///
        subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
        ylabel(.4 "40%" .5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
        xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
        xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small)) ///
	note("`notes'")
    graph export "$outdir/margins/pooled/hpiq`incl_hpiq'_ctrl`incl_ctrl'.png", replace
}

loc byvar hpi_quartile
foreach config in "110" "111"{ //by quartile
    loc incl_hpiq = substr("`config'", 1, 1)
    loc incl_ctrl = substr("`config'", 3, 1)
    if `incl_ctrl'==1{
    	loc notes "Notes: Including HPI quartile controls."
    }
    else {
    	loc notes "Notes: Including demographic and HPI quartile controls."
    }

    use "$datadir/Analysis/Demand/marg_`config'.dta", clear
	twoway ///
	rarea share_lb share_ub logdist_m  if `byvar' == 1, color(gs14%50) || ///
	rarea share_lb share_ub logdist_m  if `byvar' == 2, color(gs14%50) || ///
	rarea share_lb share_ub logdist_m  if `byvar' == 3, color(gs14%50) || ///
	rarea share_lb share_ub logdist_m  if `byvar' == 4, color(gs14%50) || ///
	line share_i logdist_m if `byvar' == 1, sort || ///
	line share_i logdist_m if `byvar' == 2, sort || ///
	line share_i logdist_m if `byvar' == 3, sort || ///
	line share_i logdist_m if `byvar' == 4, sort  ///
        graphregion(margin(b-4 l+1)) title("") ///
        legend(order(5 "HPI Quartile 1" 6 "HPI Quartile 2" 7 "HPI Quartile 3" 8 "HPI Quartile 4") size(small) region(color(none)) col(1)) ///
        subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
        ylabel(.4 "40%" .5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
        xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
        xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small)) ///
	note("`notes'")

    graph export "$outdir/margins/byqrtl/hpiq`incl_hpiq'_ctrl`incl_ctrl'.png", replace
}
