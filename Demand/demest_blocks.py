# demand estimation with block-level distance
# run after prep_block.py
import pyblp
import pandas as pd
import numpy as np

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

    # add HPI quantile to agent data
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


    # variables in the table (in order)
    tablevars = ['logdist']
    for qq in range(1, nsplits): #e.g. quantiles 1,2,3
        tablevars += [f'logdistXhpi_quantile{qq}']

    if not ref_lastq: #add last quantile only if it's not the reference for dist*hpi
        tablevars += [f'logdistXhpi_quantile{nsplits}']

    for qq in range(1, nsplits): #e.g. quantiles 1,2,3
        tablevars += [f'hpi_quantile{qq}']

    tablevars = tablevars + controls + ['1']

    # for coefficients table
    coefrows = []
    serows = []

    varlabels = []
    for vv in tablevars:
        if vv == '1':
            vv_fmt = 'Constant'
        else:
            vv_fmt = vv
        vv_fmt = vv_fmt.replace('_', ' ')
        vv_fmt = vv_fmt.replace('Xhpi', '*hpi')
        vv_fmt = vv_fmt.replace('.0]', '').replace('[', '')
        vv_fmt = vv_fmt.replace('logdist', 'log(distance)')
        vv_fmt = vv_fmt.replace('medianhhincome', 'Median Household Income')
        vv_fmt = vv_fmt.replace('medianhomevalue', 'Median Home Value')
        vv_fmt = vv_fmt.replace('popdensity', 'Population Density')
        vv_fmt = vv_fmt.replace('collegegrad', 'College Grad')
        vv_fmt = vv_fmt.title()
        vv_fmt = vv_fmt.replace('Hpi', 'HPI')

        varlabels.append(vv_fmt)
        if vv.startswith('hpi_quantile'):
            coefrows.append(f"{vv_fmt} ")
        elif 'dist' in vv:
            coefrows.append(f"{vv_fmt}" + "$^{\\dag}$ ")
        else:
            coefrows.append(f"{vv_fmt} ")
        serows.append(" ")


    # Iteration and Optimization Configurations for PyBLP
    iteration_config = pyblp.Iteration(method='lm')
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10})

    config = [True, True, False] #TODO: remove, for testing only

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
        for (ii,vv) in enumerate(agent_vars):
            print(f"demand_instruments{ii}: {vv}")
            ivcol = pd.DataFrame({f'demand_instruments{ii}': agent_data.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
            df = df.merge(ivcol, left_on='market_ids', right_index=True)

        # Solve problem
        problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)
        with pyblp.parallel(poolnum): 
            results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)
        
        # Save results
        results.to_pickle(f"{datadir}/Analysis/Demand/demest_results_{config_tag}_{setting_tag}.pkl")

        results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{config_tag}_{setting_tag}.pkl")
        # Coefficients table
        betas = results.beta.flatten()
        betases = results.beta_se.flatten()
        betalabs = results.beta_labels
        pis = results.pi.flatten()
        pises = results.pi_se.flatten()
        pilabs = results.pi_labels

        for (ii,vv) in enumerate(tablevars):
            # print(f"ii={ii}, vv={vv}")
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


        # Save utilities 
        deltas = results.compute_delta(market_id = df['market_ids'])
        deltas_df = pd.DataFrame({'market_ids': df['market_ids'], 'delta': deltas.flatten()})
        # compute block-level utilities: dot product of agent_vars and pis
        pilabs == agent_vars
        print(f"pi labels: {pilabs}")

        agent_utils = agent_data[['blkid', 'market_ids', 'hpi_quantile', 'logdist']].assign(
            agent_utility = 0,
            distcoef = 0
        )

        for (ii,vv) in enumerate(pilabs):
            print(f"ii={ii}, vv={vv}")
            coef = pis[ii]
            if 'dist' in vv:
                print(f"{vv} is a distance term, omitting from ABD and adding to coefficients instead")
                if vv=='logdist':
                    deltas_df = deltas_df.assign(distcoef = agent_data[vv])
                elif vv.startswith('logdistXhpi_quantile'):
                    qq = int(vv[-1]) #last character is the quantile number
                    agent_utils.loc[:, 'distcoef'] +=  agent_data[f"hpi_quantile{qq}"] * coef
            else:
                print(f"Adding {vv} to agent-level utility")
                agent_utils.loc[:, 'agent_utility'] += agent_data[vv] * coef

        agent_utils = agent_utils.merge(deltas_df, on='market_ids')
        agent_utils = agent_utils.assign(abd = agent_utils['agent_utility'] + agent_utils['delta'])

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