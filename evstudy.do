gl datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
gl outdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Output"
gl logdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Logs"

cap log close
log using $logdir/evstudy.log , replace

use $datadir/Analysis/panel.dta, replace


gl yvar newvax_shareunvax

//TWFE

//simple ATT
reghdfe $yvar after [aweight=unvax_pop], absorb(zip week) vce(cluster zip)

//treatment intensity by distance 

reghdfe $yvar after_* [aweight=unvax_pop], absorb(zip week) vce(cluster zip)


//event study
reghdfe $yvar ieventtime* [aweight=unvax_pop], absorb(zip week) vce(cluster zip)

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


/*
//CSDID
// simple ATT
csdid newvax_shareunvax [weight=unvax_pop], ivar(zip) time(week) gvar(treatpd) agg(simple)


// event study
csdid newvax_shareunvax [weight=unvax_pop], ivar(zip) time(week) gvar(treatpd) agg(event)
mat coef =  e(b)[1,"Tm12".."Tp12"]
mat varmat = e(V)["Tm12".."Tp12", "Tm12".."Tp12"]


gl xlabels 1 "-12" 7 "-6" 13 "0" 19 "6" 25 "12"
coefplot (matrix(coef), lwidth(thick) recast(line) v(varmat) ciopt(recast(rarea) color(navy%15)) axis(1)), ///
	vertical nooffsets legend(off) ///
	yline(0, lcolor(gray%50)) ///
	xline(13, lcolor(gray) lpattern(dash)) ///
	xlab($xlabels) xtitle("Weeks Since Treatment") title("CSDID")

graph display
graph export  "$outdir/vaxcoefplot_csdid.pdf", replace


*/



cap log close



