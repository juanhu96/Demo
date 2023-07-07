# alternative to demest_tracts.py that moves controls to tracts (agent_data)
# Demand estimation with tract-based distances.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"

#TODO: switches
hpi_quantile_in_tract = True #If True, include HPI quantile dummies in tract-level controls. If False, include them in zip-level controls. Importantly, if False, tract-level HPI*dist term must take on ZIP-level HPI quantile values.
tighter_tols = False
save_to_pipeline = False 
ref_lastq = False
nsplits = 2 #number of HPI quantiles to split the data into : 4 or 2
if not ref_lastq or not tighter_tols:
    save_to_pipeline = False
setting_tag = f"{int(bool(hpi_quantile_in_tract))}{int(bool(ref_lastq))}{int(bool(tighter_tols))}{nsplits}"
###


# data
df_read = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
agent_data_read = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")
df_read.columns
agent_data_read.columns

df_read['hpi_quantile'] = pd.qcut(df_read['hpi'], nsplits, labels=False) + 1
df_read['hpi_quantile'].value_counts()

agent_data_read['hpi_quantile'] = pd.qcut(agent_data_read['hpi'], nsplits, labels=False) + 1
agent_data_read['hpi_quantile'].value_counts()

# add ZIP-level HPI quantile to agent_data_read
ziphpiquantile = df_read[['market_ids', 'hpi_quantile']].drop_duplicates().rename(columns={'hpi_quantile': 'zip_hpi_quantile'})
agent_data_read = agent_data_read.merge(ziphpiquantile, on='market_ids')


for qq in range(1, nsplits+1):
    df_read[f'hpi_quantile{qq}'] = (df_read['hpi_quantile'] == qq).astype(int)
    if hpi_quantile_in_tract:
        agent_data_read[f'hpi_quantile{qq}'] = (agent_data_read['hpi_quantile'] == qq).astype(int)
    else:
        agent_data_read[f'hpi_quantile{qq}'] = (agent_data_read['zip_hpi_quantile'] == qq).astype(int)
    agent_data_read[f'logdistXhpi_quantile{qq}'] = agent_data_read[f'logdist'] * agent_data_read[f'hpi_quantile{qq}']


# full list of controls
controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
    'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
    'medianhomevalue', 'popdensity'] #, 'population', 'dshare']


# variables in the table (in order)
tablevars = ['logdist']
for qq in range(1, nsplits):
    tablevars += [f'logdistXhpi_quantile{qq}']

if not ref_lastq:
    tablevars += [f'logdistXhpi_quantile{nsplits}']

for qq in range(1, nsplits):
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
gtol = 1e-12 if tighter_tols else 1e-8
iteration_config = pyblp.Iteration(method='lm')
optimization_config = pyblp.Optimization('trust-constr', {'gtol':gtol})


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
    pi_init = 0.001*np.ones((1,len(agent_vars)))

    # Instruments - weighted averages of tract-level variables
    for (ii,vv) in enumerate(agent_vars):
        print(f"demand_instruments{ii}: {vv}")
        ivcol = pd.DataFrame({f'demand_instruments{ii}': agent_data.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
        df = df.merge(ivcol, left_on='market_ids', right_index=True)

    # Solve problem
    problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)
    with pyblp.parallel(32):
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


    # # Save coefficients for optimization step
    # if config == [False, False, False]: #save constant and logdist coefficients
    #     m1coefs = np.array([betas[0], pis[0]])
    #     if save_to_pipeline:
    #         np.save(f'{datadir}/Analysis/m1coefs.npy', m1coefs)
    #     np.save(f'{datadir}/Analysis/m1coefs_{config_tag}_{setting_tag}.npy', m1coefs)


    # elif config == [True, True, False]: #save constant, log(dist), HPI quantile 1, HPI quantile 2, HPI quantile 3, HPI quantile 1 * log(dist), HPI quantile 2 * log(dist), HPI quantile 3 * log(dist)]
    #     m2coefs = [betas[0]]
    #     for vv in ['logdist', 'hpi_quantile1', 'hpi_quantile2', 'hpi_quantile3', 'logdistXhpi_quantile1', 'logdistXhpi_quantile2', 'logdistXhpi_quantile3']:
    #         if vv in betalabs:
    #             m2coefs.append(betas[betalabs.index(vv)])
    #         elif vv in pilabs:
    #             m2coefs.append(pis[pilabs.index(vv)])
    #         else:
    #             print(f"ERROR: {vv} not found in results")
    #             m2coefs.append(0)
    #     m2coefs = np.array(m2coefs)
    #     if save_to_pipeline:
    #         np.save(f'{datadir}/Analysis/m2coefs.npy', m2coefs)
    #     np.save(f'{datadir}/Analysis/m2coefs_{config_tag}_{setting_tag}.npy', m2coefs)



# Complete table
coefrows = [r + "\\\\ \n" for r in coefrows]
serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]

latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
for (ii,vv) in enumerate(varlabels):
    latex += coefrows[ii]
    latex += serows[ii]

latex += "\\bottomrule\n\\end{tabular}\n\n\nNote: $^{\\dag}$ indicates a variable at the tract level."
table_path = f"{outdir}/Demand/coeftable_{config_tag}_{setting_tag}.tex"
with open(table_path, "w") as f:
    print(f"Saved table at: {table_path}")
    f.write(latex)
if save_to_pipeline:
    with open(f"{outdir}/Demand/coeftable.tex", "w") as f:
        f.write(latex)

print("Done!")


# np.load(f'{datadir}/Analysis/m1coefs.npy')
# np.load(f'{datadir}/Analysis/m2coefs.npy')