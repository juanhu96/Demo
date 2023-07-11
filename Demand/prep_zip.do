////////// MAKE MAR01 FROM RAW DATA ////////////////
gl datadir "/export/storage_covidvaccine/Data"
import delim $datadir/Raw/Vaccination/covid19vaccinesbyzipcode_test.csv, clear

rename (as_of_date zip_code_tabulation_area age12_plus_population tot_population percent_of_population_fully_vaccinated percent_of_population_partially_vaccinated) (date zip pop12up population vaxfull vaxpart)

keep date zip pop12up population vaxfull vaxpart

//merge in hpi


//merge in demographics


keep if date == "2022-03-01"





tempfile rawzip
save `rawzip'


import delim $datadir/Raw/notreallyraw/MAR01.csv, clear
merge 1:1 date zip using `rawzip'


//////////
