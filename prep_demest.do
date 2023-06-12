gl datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

import delim $datadir/Raw/MAR01.csv, clear

gen shares = vaxfull
replace shares = 0.05 if shares < 0.05
replace shares = 0.95 if shares > 0.95
gen shares_out = 1 - shares // outside shares

gen logshares = log(shares)
gen logshares_out = log(shares_out)
gen delta = logshares - logshares_out
save $datadir/MAR01_vars, replace

keep zip vaxfull distnearest hpi hpiquartile shares* race_black race_asian race_hispanic race_other ///
	health_employer health_medicare health_medicaid health_other collegegrad unemployment ///
	poverty medianhhincome medianhomevalue popdensity population
	
rename distnearest dist
gen logdist = log(dist)

// reshape long shares, i(zip) j(firm_ids)
gen market_ids = zip
gen firm_ids = 1
gen prices = 0

foreach ii of numlist 1/4{
	gen distXhpi`ii' = 1
	replace distXhpi`ii' = dist if hpiquartile == `ii'
	gen logdistXhpi`ii' = 0
	replace logdistXhpi`ii' = log(dist) if hpiquartile == `ii'
}

// merge in vote shares
tempfile df
save `df'
import delim $datadir/tracts/zip_votes.csv, clear
merge 1:1 zip using `df', keep(2 3)
drop _merge

export delim $datadir/Analysis/demest_data.csv, replace



