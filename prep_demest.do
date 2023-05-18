gl datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

import delim $datadir/Raw/MAR01.csv, clear

gen shares = .
replace shares = vaxpartfull
loc lb = 0.1
loc ub = 0.9
replace shares = `lb' if shares < `lb'
replace shares = `ub' if shares > `ub'
gen shares2 = 0.95 - shares

// plain logit
gen logshares = log(shares)
gen logshares2 = log(shares2)
gen delta = logshares - logshares2
reg delta logdistnearest
reg delta hpiquartile#c.logdistnearest
reg delta hpiquartile#c.logdistnearest race_black race_asian race_hispanic race_other ///
	health_employer health_medicare health_medicaid health_other collegegrad unemployment ///
	poverty medianhhincome medianhomevalue popdensity population, robust

keep zip vaxpartfull distnearest hpiquartile shares*
rename distnearest dist
gen logdist = log(dist)

// reshape long shares, i(zip) j(firm_ids)
gen market_ids = zip
gen firm_ids = zip
gen prices = logdist


export delim $datadir/Analysis/demest_data.csv, replace



