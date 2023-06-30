# alternative to demest_tracts.py that moves controls to tracts (agent_data)
# Demand estimation with tract-based distances.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"

# data
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables

agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")

df_read = df.assign(
    hpi_quartile1 = (df['hpi_quartile'] == 1).astype(int),
    hpi_quartile2 = (df['hpi_quartile'] == 2).astype(int),
    hpi_quartile3 = (df['hpi_quartile'] == 3).astype(int),
    hpi_quartile4 = (df['hpi_quartile'] == 4).astype(int))

agent_data_read = agent_data.assign(
    hpi_quartile1 = (agent_data['hpi_quartile'] == 1).astype(int),
    hpi_quartile2 = (agent_data['hpi_quartile'] == 2).astype(int),
    hpi_quartile3 = (agent_data['hpi_quartile'] == 3).astype(int),
    hpi_quartile4 = (agent_data['hpi_quartile'] == 4).astype(int),
    logdistXhpi_quartile1 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 1),
    logdistXhpi_quartile2 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 2),
    logdistXhpi_quartile3 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 3),
    logdistXhpi_quartile4 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 4))

# full list of controls
controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
    'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
    'medianhomevalue', 'popdensity'] #, 'population', 'dshare']

#controls that we want to move to tract level (in addition to HPI quartile dummies)
# controls_tract = ['race_asian', 'health_other', 'collegegrad', 'popdensity'] 
demog_in_agent = False
controls_tract = controls.copy() if demog_in_agent else []
hpi_quartile_in_agent = False #TODO: switch

# controls_zip is the list of controls that we want to keep at the zip level
controls_zip = [c for c in controls if c not in controls_tract]

# variables in the table (in order)
tablevars = ['logdist', 'logdistXhpi_quartile1', 'logdistXhpi_quartile2', 'logdistXhpi_quartile3'] + ['hpi_quartile1', 'hpi_quartile2', 'hpi_quartile3'] + controls + ['1']

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
    if vv.startswith('hpi_quartile'):
        if hpi_quartile_in_agent:
            coefrows.append(f"{vv_fmt}" + "$^{\\dag}$ ")
        else:
            coefrows.append(f"{vv_fmt} ")
    elif vv not in controls_zip and vv!='1':
        coefrows.append(f"{vv_fmt}" + "$^{\\dag}$ ")
    else:
        coefrows.append(f"{vv_fmt} ")
    serows.append(" ")



# Iteration and Optimization Configurations for PyBLP
tighter_tols = True #TODO: switch
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

    include_hpiquartile, interact_disthpi, include_controls = config

    print(f"***********\nRunning config: include_hpiquartile={include_hpiquartile}, interact_disthpi={interact_disthpi}, include_controls={include_controls}")
    
    df = df_read.copy()
    agent_data = agent_data_read.copy()

    # formulations
    formulation1_str = "1 + prices"

    if interact_disthpi: 
        agent_formulation_str = '0 + logdistXhpi_quartile1 + logdistXhpi_quartile2 + logdistXhpi_quartile3 + logdist'
    else:
        agent_formulation_str = '0 + logdist'

    if include_hpiquartile:
        if hpi_quartile_in_agent:
            agent_formulation_str += ' + hpi_quartile1 + hpi_quartile2 + hpi_quartile3'
        else:
            formulation1_str += ' + hpi_quartile1 + hpi_quartile2 + hpi_quartile3'
        
    if include_controls:
        for vv in controls_zip:
            formulation1_str += " + " + vv
        for vv in controls_tract:
            agent_formulation_str += " + " + vv

    formulation1 = pyblp.Formulation(formulation1_str)
    formulation2 = pyblp.Formulation('1')
    agent_formulation = pyblp.Formulation(agent_formulation_str)


    # initialize pi
    print("Agent formulation: ", agent_formulation_str)
    agent_vars = agent_formulation_str.split(' + ')
    agent_vars.remove('0')
    scale_vec = np.zeros(len(agent_vars))
    init_pi = True #Whether to initialize pi with the results from zip controls
    if init_pi:
        print("Initializing pi...")
        # read the results from zip controls - and initialize with those values
        pi_init = np.array([])
        results_zip = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_111.pkl") #from demest_tracts.py
        zippilabs = results_zip.pi_labels
        zipbetalabs = results_zip.beta_labels
        zippis = results_zip.pi.flatten() 
        zipbetas = results_zip.beta.flatten()

        for (ii,vv) in enumerate(agent_vars):
            if vv in zipbetalabs:
                labs = zipbetalabs
                vals = zipbetas
            elif vv in zippilabs:
                labs = zippilabs
                vals = zippis
            else:
                print(f"ERROR: {vv} not found in zip results")
                pi_init = np.append(pi_init, 0.001)
                continue
            zipcoef = vals[labs.index(vv)]
            # scaling 
            varmean = np.mean(abs(agent_data[vv]))
            scale = round(np.log10(varmean), 0)
            if abs(scale) > 1 and 'dist' not in vv:
                print(f"NOTE: Rescaling {vv} by 10**{scale} since mean={varmean}")
                zipcoef = zipcoef * (10**scale)
                agent_data[vv] = agent_data[vv] / (10**scale)
                scale_vec[ii] = scale
            pi_init = np.append(pi_init, zipcoef) 

        pi_init = pi_init.reshape((1, -1)) #reshape to make it a 2D array
        print("scale_vec: ", scale_vec)
        print("pi_init: ", pi_init)
        print("pi_init.shape: ", pi_init.shape)
        # store scale_vec
        if include_controls:
            np.save(f'{datadir}/Intermediate/scale_vec.npy', scale_vec)
    else:
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
    results.to_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}{int(bool(controls_tract))}.pkl")


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
            scale = scale_vec[agent_vars.index(vv)] #scale to undo scaling
            coef = pis[pilabs.index(vv)] / (10**scale)
            coef_fmt = '{:.3f}'.format(coef)
            se = pises[pilabs.index(vv)] / (10**scale)
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


    # Save coefficients for optimization step
    if config == [False, False, False]: #save constant and logdist coefficients
        m1coefs = np.array(betas[0], pis[0])
        np.save(f'{datadir}/Analysis/m1coefs.npy', m1coefs)
    elif config == [True, True, False]: #save constant, log(dist), HPI quartile 1, HPI quartile 2, HPI quartile 3, HPI quartile 1 * log(dist), HPI quartile 2 * log(dist), HPI quartile 3 * log(dist)]
        m2coefs = [betas[0]]
        for vv in ['logdist', 'hpi_quartile1', 'hpi_quartile2', 'hpi_quartile3', 'logdistXhpi_quartile1', 'logdistXhpi_quartile2', 'logdistXhpi_quartile3']:
            if vv in betalabs:
                m2coefs.append(betas[betalabs.index(vv)])
            elif vv in pilabs:
                m2coefs.append(pis[pilabs.index(vv)])
            else:
                print(f"ERROR: {vv} not found in results")
                m2coefs.append(0)
        m2coefs = np.array(m2coefs)
        np.save(f'{datadir}/Analysis/m2coefs.npy', m2coefs)



# Complete table
coefrows = [r + "\\\\ \n" for r in coefrows]
serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]

latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
for (ii,vv) in enumerate(varlabels):
    latex += coefrows[ii]
    latex += serows[ii]

latex += "\\bottomrule\n\\end{tabular}\n\n\nNote: $^{\\dag}$ indicates a variable at the tract level."
with open(f"{outdir}/Demand/coeftable_tractctrl{int(bool(controls_tract))}.tex", "w") as f:
    f.write(latex)

print("Done!")