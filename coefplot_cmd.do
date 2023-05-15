
mat lis e(b)
//hardcoding the 11 and 13 for now, assuming no lag in treatment definition
mat coef =  e(b)[1,"Tm11".."Tp13"]
mat coef_pre = e(b)[1,"Tm11".."Tp0"]
mat varmat = e(V)["Tm11".."Tp13", "Tm11".."Tp13"]
mat varmat_pre = e(V)["Tm11".."Tp0", "Tm11".."Tp0"]

global xlabels 2 "-10" 7 "-5" 12 "0" 17 "5" 22 "10"
coefplot (matrix(coef), lwidth(thick) recast(line) v(varmat) ciopt(recast(rarea) color(navy%15)) axis(1)) ///
	(matrix(coef_pre), color(gray) lwidth(thick) recast(line) v(varmat_pre) ciopt(recast(rarea) color(gray%15))), ///
	vertical nooffsets legend(off) ///
	yline(0, lcolor(gray%50)) ///
	xline(12, lcolor(gray) lpattern(dash)) ///
	xlab($xlabels) xtitle("Weeks Since Treatment") title("CSDID") subtitle("Dependent variable: $yvar") 

graph display