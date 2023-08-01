# demand estimation with block-level distance
# run after prep_block.py
import pyblp
import pandas as pd
import numpy as np
try:
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils import demest_funcs as de

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"


poolnum = 32 #number of cores to use
save_to_pipeline = False #save results to pipeline directory

# settings
nsplits = 4 #number of HPI quantiles
ref_lastq = False #reference quantile is the last one (e.g. 4th quantile)
hpi_level = 'tract' #zip or tract

# for hpi_level in ['tract', 'zip']:
for hpi_level in ['tract']:
    setting_tag = f"blk_{nsplits}q{'_reflastq' if ref_lastq else ''}_{hpi_level}"

    # read in data
    df_read = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")

    df_read.columns.tolist()
    df_read['shares'].describe()


    agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv")
    agent_data_read.columns.tolist()


    # subset to ZIPs that exist in df_read
    agent_data_read = agent_data_read.loc[agent_data_read['zip'].isin(df_read['zip']), :]

    # test
    agent_data_read['logdist'].describe()
    merged = df_read.merge(agent_data_read, on='zip', how='left')
    merged['inside_pop'] = merged['population_y'] * merged['shares']
    np.sum(merged['inside_pop']) / np.sum(merged['population_y'])
    # end test

    # make HPI quantiles 
    splits = np.linspace(0, 1, nsplits+1)

    # add HPI quantile
    df_read['hpi_quantile'] = pd.cut(df_read['hpi'], splits, labels=False, include_lowest=True) + 1
    if hpi_level == 'zip':
        agent_data_read = agent_data_read.merge(df_read[['zip', 'hpi_quantile']], on='zip', how='left')
    elif hpi_level == 'tract':
        tract_hpi = pd.read_csv(f"{datadir}/Intermediate/tract_hpi_nnimpute.csv") #from prep_tracts.py
        blk_tract_cw = pd.read_csv(f"{datadir}/Intermediate/blk_tract.csv", usecols=['tract', 'blkid']) #from block_cw.py
        agent_data_read = agent_data_read.merge(blk_tract_cw, on='blkid', how='left')
        tract_hpi['hpi_quantile'] = pd.cut(tract_hpi['hpi'], splits, labels=False, include_lowest=True) + 1
        agent_data_read = agent_data_read.merge(tract_hpi[['tract', 'hpi_quantile']], on='tract', how='left')

    # assign hpi quantile dummies and interaction terms
    # TODO: can use hpi_dist_terms() from demest_funcs.py
    for qq in range(1, nsplits+1):
        print(f"Adding hpi_quantile{qq} and logdistXhpi_quantile{qq}")
        df_read[f'hpi_quantile{qq}'] = (df_read['hpi_quantile'] == qq).astype(int)
        agent_data_read[f'hpi_quantile{qq}'] = (agent_data_read['hpi_quantile'] == qq).astype(int)
        agent_data_read[f'logdistXhpi_quantile{qq}'] = agent_data_read[f'logdist'] * agent_data_read[f'hpi_quantile{qq}']


    # full list of controls
    controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
        'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
        'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
        'medianhomevalue', 'popdensity'] #, 'population', 'dshare']


    # variables in the table (in the order it appears, including controls and agent-level variables)
    tablevars = ['logdist']
    for qq in range(1, nsplits): #e.g. quantiles 1,2,3
        tablevars += [f'logdistXhpi_quantile{qq}']

    if not ref_lastq: #add last quantile only if it's not the reference for dist*hpi
        tablevars += [f'logdistXhpi_quantile{nsplits}']

    for qq in range(1, nsplits): #e.g. quantiles 1,2,3
        tablevars += [f'hpi_quantile{qq}']

    tablevars = tablevars + controls + ['1']

    # for coefficients table
    coefrows, serows, varlabels = de.start_table(tablevars)


    # Iteration and Optimization Configurations for PyBLP
    iteration_config = pyblp.Iteration(method='lm')
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10})

    config = [True, True, True] #TODO: remove, for testing only

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

        with pyblp.parallel(poolnum): 
            results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)
        
        # Save results
        results.to_pickle(f"{datadir}/Analysis/Demand/demest_results_{config_tag}_{setting_tag}.pkl")

        # results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{config_tag}_{setting_tag}.pkl")

        # Write coefficients to table column
        coefrows, serows = de.fill_table(results, coefrows, serows, tablevars)


        # Agent utilities 
        agent_utils = de.compute_abd(results, df, agent_data)

        # save abd and coefficients
        abd_path = f"{datadir}/Analysis/Demand/agent_utils_{config_tag}_{setting_tag}.csv"
        agent_utils.to_csv(abd_path, index=False)
        print(f"Saved agent-level ABD and coefficients at: {abd_path}")
        if save_to_pipeline:
            agent_utils.to_csv(f"{datadir}/Analysis/Demand/agent_utils.csv", index=False)


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



#=============TESTING================
agent_data_read.market_ids.describe()
df_read.market_ids.describe()
# number of agents per market_ids
agent_data_read.groupby('market_ids').size().describe()
agent_data_read.population.describe()