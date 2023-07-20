# alternative to demest_tracts.py that moves controls to tracts (agent_data)
# Demand estimation with tract-based distances.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"


poolnum = 32 #number of cores to use


settings = [True, False, False, 2, 'drop', 'tract']


for settings in [
    # [True, False, False, 2, 'drop', 'tract']
    # ,
    # [True, False, False, 2, 'bottom', 'tract']
    # ,
    # [True, False, False, 2, 'nearest', 'tract']
    # ,
    # [False, False, False, 2, 'bottom', 'tract']
    # ,
    # [True, False, False, 4, 'drop', 'tract']
    # ,
    # [True, False, False, 4, 'bottom', 'tract']
    # ,
    # [True, False, False, 4, 'nearest', 'tract']
    # ,
    # [False, False, False, 4, 'bottom', 'tract']
    [True, False, False, 4, 'drop', 'old']
    ,
    [True, False, False, 4, 'bottom', 'old']
    ,
    [True, False, False, 4, 'nearest', 'old']
    ,
    [False, False, False, 4, 'bottom', 'old']

]:
    
    hpi_quantile_in_tract = settings[0] #If True, include HPI quantile dummies in tract-level controls. If False, include them in zip-level controls. Importantly, if False, tract-level HPI*dist term must take on ZIP-level HPI quantile values.
    save_to_pipeline = settings[1] #If True, save tract-level ABD and coefficients to pipeline. 
    ref_lastq = settings[2] #If True, make the last quantile the reference for dist*hpi interaction terms. If False, each quantile is its own variable.
    nsplits = settings[3] #Number of quantiles to split HPI into
    impute_hpi_method = settings[4] # 'drop' or 'bottom' or 'nearest'
    pop_method = settings[5] # 'tract' or 'zip' or 'old'

    print(f"***********\nRunning settings: hpi_quantile_in_tract={hpi_quantile_in_tract}, save_to_pipeline={save_to_pipeline}, ref_lastq={ref_lastq}, nsplits={nsplits}, impute_hpi_method={impute_hpi_method}, pop_method={pop_method}")

    setting_tag = f"{int(bool(hpi_quantile_in_tract))}{int(bool(ref_lastq))}{nsplits}{impute_hpi_method}{pop_method}"

    # data
    df_read = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
    agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data_{impute_hpi_method}_{pop_method}.csv")

    splits = np.linspace(0, 1, nsplits+1)
    df_read['hpi_quantile'] = pd.cut(df_read['hpi'], splits, labels=False, include_lowest=True) + 1

    # add HPI quantile to agent_data_read (depending on whether it's in tract or zip)
    tract_hpi = agent_data_read[['tract', 'hpi']].drop_duplicates()
    tract_hpi['hpi_quantile'] = pd.qcut(tract_hpi['hpi'], nsplits, labels=False) + 1
    tract_hpi = tract_hpi.drop(columns=['hpi'])
    ziphpiquantile = df_read[['market_ids', 'hpi_quantile']]


    if hpi_quantile_in_tract:
        agent_data_read = agent_data_read.merge(tract_hpi, on='tract')
    else:
        agent_data_read = agent_data_read.merge(ziphpiquantile, on='market_ids') 
        # agent_data_read = agent_data_read.merge(tract_hpi, on='tract') #TODO: this is just a temporary thing get HPI quantile not as a RC, but HPI*dist using tract-level HPI quantile



    # assign hpi quantile dummies and interaction terms
    for qq in range(1, nsplits+1):
        print(f"Adding hpi_quantile{qq} and logdistXhpi_quantile{qq}")
        df_read[f'hpi_quantile{qq}'] = (df_read['hpi_quantile'] == qq).astype(int)
        agent_data_read[f'hpi_quantile{qq}'] = (agent_data_read['hpi_quantile'] == qq).astype(int)
        agent_data_read[f'logdistXhpi_quantile{qq}'] = agent_data_read[f'logdist'] * agent_data_read[f'hpi_quantile{qq}']

    pd.options.display.max_columns = None
    agent_data_read.describe()
    agent_data_read.isna().sum()
    agent_data_read.hpi_quantile.value_counts()

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
            if hpi_quantile_in_tract:
                coefrows.append(f"{vv_fmt}" + "$^{\\dag}$ ")
            else:
                coefrows.append(f"{vv_fmt} ")
        elif 'dist' in vv:
            coefrows.append(f"{vv_fmt}" + "$^{\\dag}$ ")
        else:
            coefrows.append(f"{vv_fmt} ")
        serows.append(" ")



    # Iteration and Optimization Configurations for PyBLP
    iteration_config = pyblp.Iteration(method='lm')
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10})


    # config = [False, False, False]
    # config = [True, False, False]
    # config = [True, True, False]
    # config = [True, True, True]


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
            if hpi_quantile_in_tract:
                for qq in range(1, nsplits):
                    agent_formulation_str += f' + hpi_quantile{qq}'
            else:
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

        # Instruments - weighted averages of tract-level variables
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
        if config == [True, True, True]:

            deltas = results.compute_delta(market_id = df['market_ids'])
            deltas_df = pd.DataFrame({'market_ids': df['market_ids'], 'delta': deltas.flatten()})
            # compute tract-level utilities: dot product of agent_vars and pis
            pilabs == agent_vars

            tract_utils = agent_data[['tract', 'market_ids', 'hpi_quantile', 'logdist']].assign(
                tract_utility = 0,
                distcoef = 0
            )

            for (ii,vv) in enumerate(pilabs):
                coef = pis[ii]
                if 'dist' in vv:
                    print(f"{vv} is a distance term, omitting from ABD and adding to coefficients instead")
                    if vv=='logdist':
                        deltas_df = deltas_df.assign(distcoef = agent_data[vv])
                    elif vv.startswith('logdistXhpi_quantile'):
                        qq = int(vv[-1])
                        tract_utils.loc[:, 'distcoef'] +=  agent_data[f"hpi_quantile{qq}"] * coef

                else:
                    print(f"Adding {vv} to tract-level utility")
                    tract_utils.loc[:, 'tract_utility'] += agent_data[vv] * coef

            tract_utils = tract_utils.merge(deltas_df, on='market_ids')
            tract_utils = tract_utils.assign(abd = tract_utils['tract_utility'] + tract_utils['delta'])

            # save abd and coefficients
            abd_path = f"{datadir}/Analysis/Demand/tract_utils_{config_tag}_{setting_tag}.csv"
            tract_utils.to_csv(abd_path, index=False)
            print(f"Saved tract-level ABD and coefficients at: {abd_path}")
            if save_to_pipeline:
                tract_utils.to_csv(f"{datadir}/Analysis/Demand/tract_utils.csv", index=False)



    # Complete table
    coefrows = [r + "\\\\ \n" for r in coefrows]
    serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]

    latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
    for (ii,vv) in enumerate(varlabels):
        latex += coefrows[ii]
        latex += serows[ii]

    latex += "\\bottomrule\n\\end{tabular}\n\n\nNote: $^{\\dag}$ indicates a variable at the tract level."
    table_path = f"{outdir}/Demand/coeftable_{setting_tag}.tex" 
    with open(table_path, "w") as f:
        print(f"Saved table at: {table_path}")
        f.write(latex)
    if save_to_pipeline:
        with open(f"{outdir}/Demand/coeftable.tex", "w") as f:
            f.write(latex)

    print("Done!")


# np.load(f'{datadir}/Analysis/m1coefs.npy')
# np.load(f'{datadir}/Analysis/m2coefs.npy')

pd.read_csv("/export/storage_covidvaccine/Data/Analysis/Demand/tract_utils.csv")