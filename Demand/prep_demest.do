gl datadir "/export/storage_covidvaccine/Data"

import delim $datadir/Raw/notreallyraw/MAR01.csv, clear

gen shares = vaxfull
replace shares = 0.05 if shares < 0.05
replace shares = 0.95 if shares > 0.95
gen shares_out = 1 - shares // outside shares

gen logshares = log(shares)
gen logshares_out = log(shares_out)
gen delta = logshares - logshares_out
save $datadir/Analysis/Demand/MAR01_vars, replace

keep zip vaxfull distnearest hpi hpiquartile shares* race_black race_asian race_hispanic race_other ///
	health_employer health_medicare health_medicaid health_other collegegrad unemployment ///
	poverty medianhhincome medianhomevalue popdensity population
	
rename distnearest dist
gen logdist = log(dist)

// reshape long shares, i(zip) j(firm_ids)
gen market_ids = zip
gen firm_ids = 1
gen prices = 0


// merge in vote shares
tempfile df
save `df'
import delim $datadir/Intermediate/zip_votes.csv, clear
keep zip dshare
merge 1:1 zip using `df', keep(2 3)
summ dshare, d
replace dshare = r(mean) if dshare == .
drop _merge

// // merge in HPI percentile rank
// tempfile df
// save `df'
// import delim $datadir/Raw/hpi2score_zip_2011.csv, clear
// keep geoid value percentile
// rename (geoid value percentile) (zip hpi_val hpi_percentile)
// merge 1:1 zip using `df', keep(2 3)
// compare hpi_val hpidrop _merge
// // TODO: figure out right HPI to construct quantiles

export delim $datadir/Analysis/Demand/demest_data_deprecated.csv, replace



