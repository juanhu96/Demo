// Matched analysis based on pre-opening vax rate (for every treated zip, match to 3-5 that had the closest vax rate in the week before treatment) and hpiquartile (exact match). Maybe also pre-treatment distance. See PE paper - cohort#time and facility#cohort FEs. 

cap log close
log using $logdir/prep_match.log , replace



global num_neighbors_init 5
global match_vars unvax_pct
global ematch_vars hpiquartile week

use zip week treated treated_thisweek $match_vars $ematch_vars using $datadir/Analysis/Demand/panel_toreg.dta, clear

bys zip (week): gen weekb4treatment = f.treated_thisweek
replace weekb4treatment = 0 if missing(weekb4treatment)

gen const_one = 1 //Use a constant for y variable since T-effects wants a y-variable.
set seed 1 
teffects nnmatch (const_one $match_vars) (treated), ///
    atet nneighbor($num_neighbors_init) generate(nnstub) ///
    ematch($ematch_vars) osample(overlap) tlevel(1) ///
    control(0) caliper(1) metric(euclidean)

*Get the values for the matched controls.
qui forvalues nn=1/$num_neighbors_init{
    gen matching`nn' = zip[nnstub`nn'] if nnstub`nn'!=.
    foreach vv in $match_vars{
        gen `vv'_control`nn' = `vv'[nnstub`nn'] if nnstub`nn'!=.
    }
}
label values matching* zip
count if missing(matching1)
keep if !missing(matching1) //Only keep treated obs that had at least 1 match.
keep if weekb4treatment	
keep zip week matching* $match_vars *_control*

rename zip matching0
drop $match_vars *_control*

gen match_group = _n 
reshape long matching, i(match_group) j(match_rank)
rename matching zip
drop if zip==.

bysort match_group: gen n_match = _N
summ n_match

keep zip match_group

save $datadir/Intermediate/matching, replace
use $datadir/Intermediate/matching, clear


joinby zip using $datadir/Analysis/Demand/panel_toreg.dta

//fill in event time variables for control group
gsort match_group week -treated

by match_group week: replace eventtime = eventtime[1]

save $datadir/Intermediate/matched_panel.dta, replace

// gsort match_group zip week 
// br
cap log close
