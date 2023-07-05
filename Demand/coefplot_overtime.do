gl outdir "/export/storage_covidvaccine/Result"
import delimited "${outdir}/Demand/overtime/demest_coefs_control.csv", clear

rename date date_str
gen date = date(date_str, "YMD")
format date %td
tsset date

global frpp_launch_date = date("2021-02-02", "YMD")

global qcolors "black red green blue"

twoway  ///
tsline coef1 coef2 coef3 coef4, lcolor($qcolors) , ///
	legend(order(4 "HPI Quartile 4" 3 "HPI Quartile 3" 2 "HPI Quartile 2" 1 "HPI Quartile 1") region(lc(none) fc(none))) ///
	xtitle("Week") tlabel(, format(%tddmy)) ///
	ytitle("Distance coefficient") ///
	xline($frpp_launch_date, lpattern(dash)) ///
	yline(0, lpattern(dash)) ///
	note("Note: including demographic controls. Vertical line represents launch of FRPP on 2 Feb 2021.")

graph export "${outdir}/Demand/overtime/coefplot_control.png", replace

 
//   WITH SE BANDS 
 
 foreach qq of numlist 1/4{
    gen coef`qq'_lower = coef`qq' - (1.96*se`qq')
    gen coef`qq'_upper = coef`qq' + (1.96*se`qq')
}



foreach qq of numlist 1/4{
	loc qcol :word `qq' of $qcolors
	di "`qcol'"
	twoway rarea coef`qq'_lower coef`qq'_upper date, color(`qcol'%30) || ///
	tsline coef`qq', lcolor(`qcol') , ///
	 legend(order(1 "HPI Quartile `qq'") region(lc(none) fc(none))) ///
	 xtitle("Week") tlabel(, format(%tddmy)) ///
	 ytitle("Distance coefficient") ///
	 xline($frpp_launch_date, lpattern(dash)) ///
	 yline(0, lpattern(dash))
	graph export "${outdir}/Demand/overtime/coefplot_control_q`qq'.png", replace
}


/*
twoway  ///
rarea coef1_lower coef1_upper date, color(black%30) || ///
rarea coef2_lower coef2_upper date, color(red%30) || ///
rarea coef3_lower coef3_upper date, color(green%30) || ///
rarea coef4_lower coef4_upper date, color(blue%30)  || ///
tsline coef1 coef2 coef3 coef4, lcolor(black red green blue) , ///
 legend(order(4 "HPI Quartile 4" 3 "HPI Quartile 3" 2 "HPI Quartile 2" 1 "HPI Quartile 1") region(lc(none) fc(none))) ///
 xtitle("Week") tlabel(, format(%tddmy)) ///
 ytitle("Distance coefficient") ///
 xline($frpp_launch_date, lpattern(dash)) ///
 yline(0, lpattern(dash))

 
twoway  ///
rarea coef1_lower coef1_upper date, color(black%30) || ///
rarea coef2_lower coef2_upper date, color(red%30) || ///
rarea coef3_lower coef3_upper date, color(green%30) || ///
rarea coef4_lower coef4_upper date, color(blue%30)  , ///
 legend(order(4 "HPI Quartile 4" 3 "HPI Quartile 3" 2 "HPI Quartile 2" 1 "HPI Quartile 1") region(lc(none) fc(none))) ///
 xtitle("Week") tlabel(, format(%tddmy)) ///
 ytitle("Distance coefficient") ///
 xline($frpp_launch_date, lpattern(dash)) ///
 yline(0, lpattern(dash))

 */
    

