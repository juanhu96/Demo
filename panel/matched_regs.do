
global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
global outdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Output"
global logdir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Logs"
cap log close
log using $logdir/matched_regs.log , replace
qui do "/mnt/staff/zhli/VaxDemandDistance/coefplot_cmd.do"

use $datadir/Intermediate/matched_panel.dta, clear

global yvar partfull 

reghdfe $yvar ieventtime* [aweight=population12up], absorb(week#hpiquartile zip#match_group week#match_group) vce(cluster zip)
coefplot_cmd, regtype("twfe") outfile("$outdir/$yvar/coefplot/twfe_matched.pdf") note("Matched sample. Using 30km as the threshold for treatment.")


cap log close

