//trying to replicate margins
global datadir "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

************************** simplest case

use $datadir/MAR01_vars, clear

reg vaxfull logdistnearest
mat V = e(V)
mat lis V

//v is the derivative of the predictions wrt beta at x, which is just X

mat v = (2 \ 1) //this works for: margins, at(logdistnearest=(2))


qui summ logdistnearest, d
mat v = (r(mean)\ 1) //this works for: margins


mat v = (-1 \ 1), (2 \ 1) //this works for: margins, at(logdistnearest=(-1 2)) 

mat lis v
mat s = v'*V*v
mat lis s // this equals the r(V) returned by margins

margins
margins, at(logdistnearest=(2))
margins, at(logdistnearest=(-1 2)) 

mat lis r(V)


**************************add controls

use $datadir/MAR01_vars, clear

reg vaxfull logdistnearest collegegrad
mat V = e(V)
mat lis V

//v is the derivative of the predictions wrt beta at x, which is just X

qui summ collegegrad
gl meancoll = r(mean)

qui summ logdistnearest, d
gl meandist = r(mean)

mat v = ($meandist \ $meancoll \ 1) //this works for: margins


mat v = (-1 \ $meancoll \ 1),  (2 \ $meancoll \ 1)   //this works for: margins, at(logdistnearest=(-1 2)) 

mat lis v
mat s = v'*V*v
mat lis s // this equals the r(V) returned by margins

margins
margins, at(logdistnearest=(2))
margins, at(logdistnearest=(-1 2)) 

mat lis r(V)

**************************non-linear transformation
use $datadir/MAR01_vars, clear

gen vax01 = vaxfull>.6
tab vax01
logit vax01 logdistnearest, noconst
mat lis e(b)
mat V = e(V)
mat lis V

//v is the derivative of the outcome of interest wrt beta at x, which is...
//by chain rule: x * d

gl p = invlogit(_b["logdistnearest"]*2)
gl d = $p * (1-$p)

mat v = ($d * 2) // works for: margins, at(logdistnearest=(2))

mat lis v
mat s = v'*V*v
mat lis s // this equals the r(V) returned by margins

margins, at(logdistnearest=(2))

mat lis r(V)

************************** logit without demographics
// dsdb = dsdv * dvdb
// b is pi
// d is logdist
// s is share

// dsdv = d/dv exp(v)/[1+exp(v)] = exp(v)/[1+exp(v)]^2 = (s)(1-s)
// dvdb = d

// TODO: do we care about agents and weights?










