# demand estimation with capacity constraints

# 0. Draw a random order for demanders once.
# 1. Estimate the demand model under a given matching function (start with closest facility and no capacity constraints)
# 2. Apply estimated demand to the random order of demanders, except the distance for each demander is based on the closest location with available capacity by the time they get their choice.
# 3. Repeat step 1 assuming the location matching from step 2. Repeat iteratively until you reach a fixed point

# Notation: I use "geog" to denote a block/tract. We're using blocks now.

import pandas as pd
import numpy as np
import pyblp
import sys
import time

print("Entering demest_assm.py")
time_entered = time.time()


try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
    from demand_utils import fixed_point as fp
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de
    from Demand.demand_utils import fixed_point as fp

np.random.seed(1234)

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result/Demand"


#=================================================================
# SETTINGS
#=================================================================
testing = sys.argv == [''] #test if running in terminal, full run if running in shell script

testing = False #TODO: remove

capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 8000 #capacity per location. lower when testing
# capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 #capacity per location. lower when testing
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 200 #maximum rank to offer
nsplits = int(sys.argv[3]) if len(sys.argv) > 3 else 3 #number of HPI quantiles
hpi_level = sys.argv[4] if len(sys.argv) > 4 else 'zip' #zip or tract



# in rundemest_assm.sh we have, e.g.:
# nohup python3 /users/facsupport/zhli/VaxDemandDistance/Demand/demest_assm.py 10000 10 > demest_assm_10000_10.out &

only_constant = False #if True, only estimate constant term in demand. default to False
no_dist_heterogeneity = False #if True, no distance heterogeneity in demand. default to False
cap_coefs_to0 = False # if we get positive distance coefficients, set them to 0 TODO:

setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
setting_tag += "_const" if only_constant else ""
setting_tag += "_nodisthet" if no_dist_heterogeneity else ""
setting_tag += "_capcoefs0" if cap_coefs_to0 else ""
setting_tag += f"_{hpi_level}hpi" if hpi_level != 'zip' else ""

print(setting_tag)
coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs" if not testing else None


#=================================================================
# Data for assignment: distances, block-market crosswalk, population
#=================================================================

print(f"Testing: {testing}")
print(f"Capacity: {capacity}")
print(f"Max rank: {max_rank}")
print(f"Number of HPI quantiles: {nsplits}")
print(f"Setting tag: {setting_tag}")
print(f"Coef save path: {coefsavepath}")
sys.stdout.flush()

cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
blocks_unique = np.unique(cw_pop.blkid.values)
markets_unique = np.unique(cw_pop.market_ids.values)
if testing:
    test_frac = 0.05
    ngeog = len(blocks_unique)
    test_ngeog = int(round(test_frac*ngeog, 0))
    blocks_tokeep = np.random.choice(blocks_unique, size=test_ngeog, replace=False)
    capacity = capacity * test_frac  #capacity per location. lower when testing
    cw_pop = cw_pop.loc[cw_pop.blkid.isin(blocks_tokeep), :]
else:
    test_frac = 1
    blocks_tokeep = blocks_unique


cw_pop.sort_values(by=['blkid'], inplace=True)

print("Number of geogs:", cw_pop.shape[0]) # 377K
print("Number of individuals:", cw_pop.population.sum()) # 39M
sys.stdout.flush()



#=================================================================
# Data for demand estimation: market-level data, agent-level data
#=================================================================

ref_lastq = False


controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
    'collegegrad', 'unemployment', 'poverty', 'logmedianhhincome', 
    'logmedianhomevalue', 'logpopdensity']
df_colstokeep = controls + ['market_ids', 'firm_ids', 'shares', 'prices'] + ['hpi']

formulation1_str = "1 + prices"
for qq in range(1, nsplits):
    formulation1_str += f' + hpi_quantile{qq}'
for vv in controls:
    formulation1_str += " + " + vv

if only_constant:
    formulation1_str = "1 + prices"

if no_dist_heterogeneity:
    agent_formulation_str = '0 + logdist'
else:
    agent_formulation_str = '0 +'
    for qq in range(1, nsplits):
        agent_formulation_str += f' logdistXhpi_quantile{qq} +'
    if ref_lastq:
        agent_formulation_str += ' logdist'
    else:
        agent_formulation_str += f' logdistXhpi_quantile{nsplits}'

formulation1 = pyblp.Formulation(formulation1_str)
formulation2 = pyblp.Formulation('1')
product_formulations = (formulation1, formulation2)
agent_formulation = pyblp.Formulation(agent_formulation_str)

results = pd.read_pickle(f"{outdir}/results_8000_200_3q.pkl")

for nsplits in [3, 4]:
    tablevars = []
    for qq in range(1, nsplits+1): #e.g. quantiles 1,2,3
        tablevars += [f'logdistXhpi_quantile{qq}']

    for qq in range(1, nsplits): #e.g. quantiles 1,2,3
        tablevars += [f'hpi_quantile{qq}']

    tablevars = tablevars + controls + ['1']
    print("Table variables:", tablevars)

    coefrows, serows, varlabels = de.start_table(tablevars)
    for capacity in [8000,10000,12000]:
        setting_tag = f"{capacity}_200_{nsplits}q"
        results = pd.read_pickle(f"{outdir}/results_{setting_tag}.pkl")
        print(setting_tag)
        print(results)
        coefrows, serows = de.fill_table(results, coefrows, serows, tablevars)

    coefrows = [r + "\\\\ \n" for r in coefrows]
    serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]



    latex = "\\begin{tabular}{lccc}\n \\toprule\n\\midrule\n \\ & \\multicolumn{3}{c}{Capacity} \\\\ Variable & 8000 & 10000 & 12000 \\\\ \\midrule\n"
    for (ii,vv) in enumerate(varlabels):
        latex += coefrows[ii]
        latex += serows[ii]
    latex += "\\bottomrule\n\\end{tabular}\n\n"

    table_path = f"{outdir}/coeftables/coeftable_{nsplits}q.tex"

    with open(table_path, "w") as f:
        print(f"Saved table at: {table_path}")
        f.write(latex)
