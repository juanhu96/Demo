# Coefficient table after demest_tracts.py

import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"


vars = 'logdist'
vars += ' + hpi_quartile1 + hpi_quartile2 + hpi_quartile3'
vars += ' + '
vars += ' + '.join(['race_black', 'race_asian', 'race_hispanic', 'race_other', 'health_employer', 'health_medicare', 'health_medicaid', 'health_other', 'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 'medianhomevalue', 'popdensity'])
vars += ' + hpi_quartile[1.0]*logdist + hpi_quartile[2.0]*logdist + hpi_quartile[3.0]*logdist'
vars += ' + 1'

# NOTE: `vars` should match variable names in the pyblp formulation, `vv_fmt` is the formatted version for the table

coefrows = []
serows = []

varlabels = []
for vv in vars.split(' + '):
    if vv == '1':
        vv_fmt = 'Constant'
    else:
        vv_fmt = vv

    vv_fmt = vv_fmt.replace('_', ' ')
    vv_fmt = vv_fmt.replace('.0]', '').replace('[', '')
    vv_fmt = vv_fmt.replace('logdist', 'log(distance)')
    vv_fmt = vv_fmt.replace('medianhhincome', 'Median Household Income')
    vv_fmt = vv_fmt.replace('medianhomevalue', 'Median Home Value')
    vv_fmt = vv_fmt.replace('popdensity', 'Population Density')
    vv_fmt = vv_fmt.replace('collegegrad', 'College Grad')
    vv_fmt = vv_fmt.title()
    vv_fmt = vv_fmt.replace('Hpi', 'HPI')
    varlabels.append(vv_fmt)
    coefrows.append(f"{vv_fmt} ")
    serows.append(" ")



# config = [False, False, False]
# config = [True, False, False]
# config = [True, True, False]
# config = [True, True, True]


for config in [ #loop over configs (columns)
    [False, False, False],
    [True, False, False],
    [True, True, False],
    [True, True, True]
    ]:

    include_hpiquartile, interact_disthpi, include_controls = config

    print(f"config: include_hpiquartile={include_hpiquartile}, interact_disthpi={interact_disthpi}, include_controls={include_controls}")

    results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")

    betas = results.beta.flatten()
    betases = results.beta_se.flatten()
    betalabs = results.beta_labels
    pis = results.pi.flatten()
    pises = results.pi_se.flatten()
    pilabs = results.pi_labels

    for (ii,vv) in enumerate(vars.split(' + ')):
        if interact_disthpi and vv == 'logdist':
            vv = 'hpi_quartile[4.0]*logdist'

        if vv in betalabs:
            coef = betas[betalabs.index(vv)]
            coef_fmt = '{:.3f}'.format(coef)
            se = betases[betalabs.index(vv)]
            se_fmt = '(' + '{:.3f}'.format(se) + ')'
        elif vv in pilabs:
            coef = pis[pilabs.index(vv)]
            coef_fmt = '{:.3f}'.format(coef)
            se = pises[pilabs.index(vv)]
            se_fmt = '(' + '{:.3f}'.format(se) + ')'
        else: #empty cell if vv is not used in this config
            coef = 0
            coef_fmt = ''
            se = np.inf
            se_fmt = ''

        # add significance stars 
        if abs(coef/se) > 2.576:
            coef_fmt += '$^{***}$'
        elif abs(coef/se) > 1.96:
            coef_fmt += '$^{**}$'
        elif abs(coef/se) > 1.645:
            coef_fmt += '$^{*}$'

        # append to existing rows
        coefrows[ii] += f"& {coef_fmt}"
        serows[ii] += f"& {se_fmt}"

        
coefrows = [r + "\\\\ \n" for r in coefrows]
serows = [r + "\\\\ \n" for r in serows]

latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
for (ii,vv) in enumerate(varlabels):
    latex += coefrows[ii]
    latex += serows[ii]

latex += "\\bottomrule\n\\end{tabular}\n"
with open(f"{outdir}/Demand/coeftable.tex", "w") as f:
    f.write(latex)
