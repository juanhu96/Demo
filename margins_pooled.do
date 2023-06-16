global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
global outdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Output"

use "$datadir/Analysis/tracts_marg_pooled_000.dta", clear

twoway line share_i logdist_m, sort || ///
line share_lb logdist_m, sort || ///
line share_ub logdist_m, sort   ///
    graphregion(margin(b-4 l+1)) title("") ///
    legend(size(small) region(color(none))) ///
    subtitle("Adjusted Vaccination" "Coverage", size(small) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
    ylabel(.4 "40%" .5 "50%" .6 "60%" .7 "70%" .8 "80%" .9 "90%", labsize(small)) ytitle("") ///
    xtitle("Kilometers to Nearest Vaccination Site (log scale)", size(small)) ///
    xlabel(-.5 ".6" 0 "1" .693 "2" 1.609 "5" 2.303 "10", labsize(small))	
// graph export "$outdir/margins/pooled_se.png", replace


