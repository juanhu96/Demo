// NOTE: rename the input data's latitude and longitude variables to "latitude" and "longitude" before running this script.
cap log close
log using "geonear.log", replace

*Pass in parameters.
local baselocs "`1'"
local nborlocs "`2'"
local outpath "`3'"
local nearcount "`4'"
if "`nearcount'" == "" local nearcount 1

use "`baselocs'", clear

ds

geonear id latitude longitude using "`nborlocs'", neighbors(id latitude longitude) report(5) nearcount(`nearcount')

export delim "`outpath'", replace

cap log close
