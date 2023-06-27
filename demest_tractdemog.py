# alternative to demest_tracts.py that moves controls to tracts (agent_data)
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"

config = [True, True, True]

include_hpiquartile, interact_disthpi, include_controls = config

print(f"Running config: include_hpiquartile={include_hpiquartile}, interact_disthpi={interact_disthpi}, include_controls={include_controls}")
 
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables

agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
    'health_employer', 'health_medicare', 'health_medicaid', 'health_other', 'collegegrad',
    'unemployment', 'poverty', 'medianhhincome', 
    'medianhomevalue', 'popdensity'] #this is the same controls vector at ZIP level

#controls that we want to move to tract level TODO: change this    
controls_tract = ['collegegrad', 'race_asian', 'health_other', 'popdensity'] #works
# controls_tract = controls.copy()


# read the results from zip controls - and initialize with those values
results_zip = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_111.pkl")
pi_init = results_zip.pi.flatten() #start with the logdistXhpi_quartile coefficients from ZIP-level estimation
for (ii, vv) in enumerate(controls_tract):
    vv_position_in_beta = results_zip.beta_labels.index(vv)
    zip_beta = results_zip.beta[vv_position_in_beta]
    # rescale if necessary for optimization performance
    while np.mean(abs(agent_data[vv])) > 1:
        print(f"rescaling {vv} by 10\n   since mean={np.mean(abs(agent_data[vv]))}")
        zip_beta = zip_beta * 10
        agent_data[vv] = agent_data[vv] / 10
    pi_init = np.append(pi_init, zip_beta) 

pi_init = pi_init.reshape((1, -1)) #reshape to make it a 2D array

initialize_with_zeros = True
if initialize_with_zeros:
    pi_init = 0.001* np.ones((1, pi_init.shape[1])) #initialize with near-zero values to avoid bias from ZIP-level estimates
    print(pi_init)




for cc in controls_tract:
    controls.remove(cc)



print(controls)
formula_str = "1 + prices +  " + " + ".join(controls)




hpi_in_tract = True
if hpi_in_tract:
    agent_data = agent_data.assign(
        hpi_quartile1 = (agent_data['hpi_quartile'] == 1).astype(int),
        hpi_quartile2 = (agent_data['hpi_quartile'] == 2).astype(int),
        hpi_quartile3 = (agent_data['hpi_quartile'] == 3).astype(int))
    controls_tract = controls_tract + ['hpi_quartile1', 'hpi_quartile2', 'hpi_quartile3']
    formulation1 = pyblp.Formulation(formula_str)
    pi_init = np.concatenate((pi_init, 0.001*np.ones((1,3))), axis=1) 
else:
    df = df.assign(
        hpi_quartile1 = (df['hpi_quartile'] == 1).astype(int),
        hpi_quartile2 = (df['hpi_quartile'] == 2).astype(int),
        hpi_quartile3 = (df['hpi_quartile'] == 3).astype(int))
    formulation1 = pyblp.Formulation(formula_str + '+ hpi_quartile1 + hpi_quartile2 + hpi_quartile3')
    
formulation2 = pyblp.Formulation('1')
agent_formulation = pyblp.Formulation('0 + logdist:C(hpi_quartile)' + ' + ' + ' + '.join(controls_tract))

# make instruments
# (a) distance * hpi_quartile
for qq in range(1,5):
    agent_data[f"logdistXhpi{qq}"] = agent_data['logdist'] * (agent_data['hpi_quartile'] == qq)
    demand_instruments_qq = pd.DataFrame({f'demand_instruments{qq-1}': agent_data.groupby('market_ids')[f'logdistXhpi{qq}'].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
    df = df.merge(demand_instruments_qq, left_on='market_ids', right_index=True)
# (b) demographic controls we moved to tract level
for (ii,vv) in enumerate(controls_tract):
    demand_instruments_vv = pd.DataFrame({f'demand_instruments{qq+ii}': agent_data.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
    df = df.merge(demand_instruments_vv, left_on='market_ids', right_index=True)

# pd.set_option('display.max_columns', None)

problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)

tighter_tols = False
if tighter_tols:
    iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12})
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-10, 'verbose':0})
else:
    iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-11})
    optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-8, 'verbose':1})

with pyblp.parallel(48):
    results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)

# print(results)
results.to_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}_tractctrl.pkl")

results = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}_tractctrl.pkl")
results0 = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")
print("*********\nResults with controls moved to tracts:")
print("NOTE: CHECK SCALING")
print(results)
print("*********\nResults with controls in ZIP:")
print(results0)

# results0noctrl = pyblp.read_pickle(f"{datadir}/Analysis/Demand/demest_results_{int(include_hpiquartile)}{int(interact_disthpi)}0.pkl")
# print("*********\nResults without controls:")
# print(results0noctrl)
