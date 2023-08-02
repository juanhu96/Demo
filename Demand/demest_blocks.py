# demand estimation with block-level distance
# run after prep_block.py
import pyblp
import pandas as pd
import numpy as np
import sys
try:
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import demest_funcs as de

pyblp.options.digits = 3
pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0 #don't check for collinearity

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"

poolnum = 1 #number of cores to use
save_to_pipeline = False #save results to pipeline directory

#### Settings
# nohup python3 /users/facsupport/zhli/VaxDemandDistance/Demand/demest_blocks.py 4 tract False > demest_blocks_4q_tract.out &

print("Running demest_blocks.py with settings:", sys.argv)
#number of HPI quantiles
nsplits = int(sys.argv[1]) if len(sys.argv) > 1 else 4

#zip or tract
hpi_level = sys.argv[2] if len(sys.argv) > 2 else 'tract'

#reference quantile is the last one (e.g. 4th quantile)
ref_lastq = False
if len(sys.argv) > 3:
    if sys.argv[3] == 'True':
        ref_lastq = True

#===================================================================================================

setting_tag = f"blk_{nsplits}q{'_reflastq' if ref_lastq else ''}_{hpi_level}"
print(f"Setting tag: {setting_tag}")
# read in data
df_read = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df_read.columns.tolist()
df_read['shares'].describe()


agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
agent_data_read.columns.tolist()

# subset to ZIPs that exist in both datasets
zips_in_both = set(df_read['zip'].tolist()).intersection(set(agent_data_read['zip'].tolist()))
print("Number of ZIPs:", len(zips_in_both))
agent_data_read = agent_data_read.loc[agent_data_read['zip'].isin(zips_in_both)]
df_read = df_read.loc[df_read['zip'].isin(zips_in_both)]

# make HPI quantiles 
splits = np.linspace(0, 1, nsplits+1)

# add HPI and HPI*distance terms to market data
df_read = de.hpi_dist_terms(df_read, nsplits=nsplits, add_bins=True, add_dummies=True, add_dist=False)

# add HPI quantile to agent data
if hpi_level == 'zip':
    agent_data_read = agent_data_read.merge(df_read[['zip', 'hpi_quantile']], on='zip', how='left')
elif hpi_level == 'tract':
    tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") #from prep_tracts.py
    blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) #from block_cw.py
    agent_data_read = agent_data_read.merge(blk_tract_cw, on='blkid', how='left')
    tract_hpi['hpi_quantile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1
    agent_data_read = agent_data_read.merge(tract_hpi[['tract', 'hpi_quantile']], on='tract', how='left')

agent_data_read = de.hpi_dist_terms(agent_data_read, nsplits=nsplits, add_bins=False, add_dummies=True, add_dist=True)


# full list of controls
controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
    'collegegrad', 'unemployment', 'poverty', 'logmedianhhincome', 
    'logmedianhomevalue', 'logpopdensity'] #, 'population', 'dshare']


# variables in the table (in the order it appears, including controls and agent-level variables)
tablevars = ['logdist']
for qq in range(1, nsplits): #e.g. quantiles 1,2,3
    tablevars += [f'logdistXhpi_quantile{qq}']

if not ref_lastq: #add last quantile only if it's not the reference for dist*hpi
    tablevars += [f'logdistXhpi_quantile{nsplits}']

for qq in range(1, nsplits): #e.g. quantiles 1,2,3
    tablevars += [f'hpi_quantile{qq}']

tablevars = tablevars + controls + ['1']
print("Table variables:", tablevars)

# for coefficients table
coefrows, serows, varlabels = de.start_table(tablevars)


# Iteration and Optimization Configurations for PyBLP
iteration_config = pyblp.Iteration(method='lm')
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10})

config = [True, True, True] #TODO:  for testing only

for config in [
    [False, False, False],
    [True, False, False],
    [True, True, False],
    [True, True, True]
    ]:

    include_hpiquantile, interact_disthpi, include_controls = config
    config_tag = f"{int(include_hpiquantile)}{int(interact_disthpi)}{int(include_controls)}"

    print(f"***********\nRunning config: include_hpiquantile={include_hpiquantile}, interact_disthpi={interact_disthpi}, include_controls={include_controls}")
    
    df = df_read.copy()
    agent_data = agent_data_read.copy()

    # formulations
    formulation1_str = "1 + prices"

    if interact_disthpi:
        agent_formulation_str = '0 +'
        for qq in range(1, nsplits):
            agent_formulation_str += f' logdistXhpi_quantile{qq} +'
        if ref_lastq:
            agent_formulation_str += ' logdist'
        else:
            agent_formulation_str += f' logdistXhpi_quantile{nsplits}'
    else:
        agent_formulation_str = '0 + logdist'

    if include_hpiquantile:
        for qq in range(1, nsplits):
            formulation1_str += f' + hpi_quantile{qq}'
        
    if include_controls:
        for vv in controls:
            formulation1_str += " + " + vv

    formulation1 = pyblp.Formulation(formulation1_str)
    formulation2 = pyblp.Formulation('1')
    agent_formulation = pyblp.Formulation(agent_formulation_str)


    # initialize pi
    print("Agent formulation: ", agent_formulation_str)
    agent_vars = agent_formulation_str.split(' + ')
    agent_vars.remove('0')
    pi_init = 0.01*np.ones((1,len(agent_vars)))

    # Instruments - weighted averages of agent-level variables
    df = de.add_ivcols(df, agent_data, agent_vars)

    # Solve problem
    problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)

    if poolnum == 1:
        results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)
    else:
        with pyblp.parallel(poolnum): 
            results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)
    
    # Save results

    # results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{config_tag}_{setting_tag}.pkl")

    # Write coefficients to table column
    coefrows, serows = de.fill_table(results, coefrows, serows, tablevars)

    # Agent utilities 
    agent_utils = de.compute_abd(results, df, agent_data)

    # save abd and coefficients
    if save_to_pipeline:
        abd_path = f"{datadir}/Analysis/Demand/agent_utils_{config_tag}_{setting_tag}.csv"
        agent_utils.to_csv(abd_path, index=False)
        agent_utils.to_csv(f"{datadir}/Analysis/Demand/agent_utils.csv", index=False)
        results.to_pickle(f"{datadir}/Analysis/Demand/demest_results_{config_tag}_{setting_tag}.pkl")
        print(f"Saved agent-level ABD and coefficients at: {abd_path}")


# Complete table
coefrows = [r + "\\\\ \n" for r in coefrows]
serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]

latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
for (ii,vv) in enumerate(varlabels):
    latex += coefrows[ii]
    latex += serows[ii]

latex += "\\bottomrule\n\\end{tabular}\n\n\nNote: $^{\\dag}$ indicates a variable at the block level."
table_path = f"{outdir}/Demand/coeftable_{setting_tag}.tex" 

with open(table_path, "w") as f:
    print(f"Saved table at: {table_path}")
    f.write(latex)

if save_to_pipeline:
    with open(f"{outdir}/Demand/coeftable.tex", "w") as f:
        f.write(latex)

print("Done!")



# #=============TESTING================
# agent_data_read.market_ids.describe()
# df_read.market_ids.describe()
# # number of agents per market_ids
# agent_data_read.groupby('market_ids').size().describe()
# agent_data_read.population.describe()