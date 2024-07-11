// run after margins.py

global datadir "/export/storage_covidvaccine/Data"
global outdir "/export/storage_covidvaccine/Result/Demand"

gl setting_tag "10000_5_4q_mnl"

use "$datadir/Analysis/Demand/marg_${setting_tag}_onlypreds.dta", clear

twoway ///
line share_i dist_m if hpi_quantile == 1, sort lwidth(0.7) color(red%80) || ///
line share_i dist_m if hpi_quantile == 2, sort lwidth(0.7) color(orange%80) || ///
line share_i dist_m if hpi_quantile == 3, sort lwidth(0.7) color(green%80) || ///
line share_i dist_m if hpi_quantile == 4, sort lwidth(0.7) color(blue%80) ///
    graphregion(margin(b-4 l+1)) title("") ///
    legend(order(8 "Top 25%" 7 "50-75%" 6 "25-50%" 5 "Bottom 25%" ) size(medium) region(color(none)) col(1) title("Healthy Places Index" "(Quartile)", size(medium))) ///
    subtitle("Vaccinated (%)", size(medium) position(11) ring(1) span margin(l=-3 b=2.5) justification(center)) ///
    ylabel(, labsize(medium)) ytitle("") ///
    xtitle("Distance to Nearest Vaccination Site (km)", size(medium)) ///
    xlabel(, labsize(medium))

graph export "$outdir//margins/margins_${setting_tag}_onlypreds.png", replace width(1600) height(1000)
