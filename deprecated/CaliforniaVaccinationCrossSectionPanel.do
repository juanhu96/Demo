

*************************************	
* Cross sectional data (pharmacies) *
*************************************
clear
import delimited "~/downloads/MAR01.csv", clear
drop v1

reg vaxfull logdistnearest, robust
est store vax

reg vaxfull logdistnearest b4.hpiquartile, robust
est store vaxhpi
	
reg vaxfull c.logdistnearest##b4.hpiquartile, robust
est store vaxinteraction

reg vaxfull c.logdistnearest##b4.hpiquartile race_black race_asian race_hispanic race_other ///
	health_employer health_medicare health_medicaid health_other collegegrad unemployment ///
	poverty medianhhincome medianhomevalue popdensity population, robust
est store vaxdemo

esttab vax* using "C:\Users\elong\Dropbox\COVID Vaccination\Vaccination Data\California\ResultsVax.csv", replace star(* 0.05 ** 0.01 *** 0.001) se scalars(df_m df_a_initial F r2 r2_a absvars)







***************************************	
* Panel data (mass vaccination sites) *
***************************************	
clear
import delimited "~/downloads/JAN_MAY.csv", clear
drop v1 x
destring race*, replace force


* Calculate % of population newly vaccinated each week 
drop if population == "NA"
destring population, replace
gen vaxnewrate = vaxnew / population

* Create lagged variable for cumulative % fully vaccinated in previous week
rename date date_str
gen date = date(date_str,"YMD")
format date %td
gen week = wofd(date)
format week %tw

gen unvax_pop = population12up - vaxfull
gen unvax_pct = unvax_pop/population12up

gen newvax_shareunvax = vaxnew/unvax_pop
replace newvax_shareunvax if newvax_shareunvax<0 | newvax_shareunvax>.3
gen asinh_newvaxshareunvax = asinh(newvax_shareunvax)

* Regressions with lagged cumulative vaccinations
reghdfe vaxnewrate logdistnearestopen lagfull [aweight=population], absorb(week zip) vce(cluster zip)
reghdfe newvax_shareunvax logdistnearestopen#race_white  [aweight=population], absorb(week zip) vce(cluster zip)



est store massvax 

esttab massvax using "C:\Users\elong\Dropbox\COVID Vaccination\Vaccination Data\California\ResultsMassVax.csv", replace star(* 0.05 ** 0.01 *** 0.001) se scalars(df_m df_a_initial F r2 r2_a absvars)
	
