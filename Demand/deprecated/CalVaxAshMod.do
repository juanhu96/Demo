

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







***************************************	
* Panel data (mass vaccination sites) *
***************************************	
clear
import delimited "~/downloads/JAN_MAY.csv", clear
drop v1 x
destring race* hpi hpiquartile, replace force


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
replace newvax_shareunvax =. if newvax_shareunvax<0 | newvax_shareunvax>.3
gen asinh_newvaxshareunvax = asinh(newvax_shareunvax)

xtset zip week


* Regressions with lagged cumulative vaccinations
reghdfe newvax_shareunvax c.logdistnearestopen#i.hpiquartile [aweight=unvax_pop] if unvax_pop>0, absorb(week zip) vce(cluster zip)

reghdfe newvax_shareunvax L.partfull c.logdistnearest##b4.hpiquartile race_black race_asian race_hispanic race_other [aweight=unvax_pop] if unvax_pop>0, absorb(week#hpiquartile zip) vce(cluster zip)




