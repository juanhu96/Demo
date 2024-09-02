// NOTE: rename the input data's latitude and longitude variables to "latitude" and "longitude" before running this script.
cap log close
// JINGYUAN: commented out log file so that we won't be writing on the same log file
// log using "geonear.log", replace

*Pass in parameters.
local baselocs "`1'"
local nborlocs "`2'"
local outpath "`3'"
local within "`4'"
local limit "`5'"

if "`within'" == "" local within 200 //max distance in km to search for neighbors
if "`limit'" == "" local limit 300 //max number of neighbors to return

di "baselocs: `baselocs'"
di "nborlocs: `nborlocs'"
di "outpath: `outpath'"
di "within: `within'"
di "limit: `limit'"

use "`baselocs'", clear

ds

geonear blkid latitude longitude using "`nborlocs'", neighbors(id latitude longitude) long within(`within') limit(`limit')

rename id locid
rename km_to_id dist

gen logdist = log(dist)

count
ds

gsort blkid logdist //this is important as future code assumes this

export delim "`outpath'", replace

cap log close
