# Demand estimation with tract-based distances, with time-varying vax rates.
# run after prep_tracts.py
import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result"

# data
panel = pd.read_csv(f"{datadir}/Raw/notreallyraw/CaliforniaVaccinationZip.csv", usecols=['Zip', 'Date', 'Pop12up', 'VaxFull'])
panel.columns = panel.columns.str.lower()
panel.dropna(inplace=True)

# winsorize shares to 0.05 and 0.95
panel['shares'] = panel['vaxfull'].clip(lower=0.05, upper=0.95)
print(panel['shares'].describe())

# format date
panel['date'] = pd.to_datetime(panel['date'], format='%m/%d/%Y')
panel.sort_values(by=['date', 'zip'], inplace=True)

dates = panel['date'].unique()
len(dates) # 67 weeks

panel['prices'] = 0
panel['market_ids'] = panel['zip']
panel['firm_ids'] = 1

# read tract data
agent_data = pd.read_csv(f"{datadir}/Analysis/Demand/agent_data.csv")
agent_data = agent_data.assign(
    hpi_quartile1 = (agent_data['hpi_quartile'] == 1).astype(int),
    hpi_quartile2 = (agent_data['hpi_quartile'] == 2).astype(int),
    hpi_quartile3 = (agent_data['hpi_quartile'] == 3).astype(int),
    hpi_quartile4 = (agent_data['hpi_quartile'] == 4).astype(int),
    logdistXhpi_quartile1 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 1),
    logdistXhpi_quartile2 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 2),
    logdistXhpi_quartile3 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 3),
    logdistXhpi_quartile4 = agent_data['logdist'] * (agent_data['hpi_quartile'] == 4))


# formulations
formulation1 = pyblp.Formulation('1 + prices')
formulation2 = pyblp.Formulation('1')
agent_formulation_str = '0 + logdistXhpi_quartile1 + logdistXhpi_quartile2 + logdistXhpi_quartile3 + logdistXhpi_quartile4'
agent_formulation = pyblp.Formulation(agent_formulation_str)
agent_vars = agent_formulation_str.split(' + ')
agent_vars.remove('0')


# instruments
for (ii,vv) in enumerate(agent_vars):
    print(f"demand_instruments{ii}: {vv}")
    ivcol = pd.DataFrame({f'demand_instruments{ii}': agent_data.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
    panel = panel.merge(ivcol, left_on='market_ids', right_index=True)



# Iteration and Optimization Configurations for PyBLP
tighter_tols = False #TODO: switch
gtol = 1e-12 if tighter_tols else 1e-8
iteration_config = pyblp.Iteration(method='lm')
optimization_config = pyblp.Optimization('trust-constr', {'gtol':gtol})

pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0


# vector to store distance coefficients
coef_vec = []
se_vec = []


# TODO: testing with fewer weeks
# dates_torun = dates[::10]
# dates_torun = dates[20:30]
dates_torun = dates


for (ii, ww) in enumerate(dates_torun):
    df = panel[panel['date'] == ww]
    
    # initial guess
    pi_init = coef_vec[-1] if ii > 0 else -0.1*np.ones((1,4)) 
    
    problem = pyblp.Problem(product_formulations=(formulation1, formulation2), product_data=df, agent_formulation=agent_formulation, agent_data=agent_data)
    with pyblp.parallel(32):
        results = problem.solve(pi=pi_init, sigma = 0, iteration = iteration_config, optimization = optimization_config)

    # save distance coefficients
    coef = results.pi
    coef_vec.append(coef)
    se = results.pi_se
    se_vec.append(se)

    print(f"Week {ii+1} of {len(dates_torun)}. Date: {ww}. Distance coefficient: {coef}")


# save distance coefficients
coef_mat = np.concatenate(coef_vec, axis=0)
se_mat = np.concatenate(se_vec, axis=0)

df = pd.DataFrame(np.concatenate([coef_mat, se_mat], axis=1))
df.columns = ['coef1', 'coef2', 'coef3', 'coef4', 'se1', 'se2', 'se3', 'se4']
df['date'] = dates_torun
df.to_csv(f"{datadir}/Result/Demand/overtime/demest_coefs.csv", index=False)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df['date'], df['coef1'], label='HPI Quartile 1')
ax.plot(df['date'], df['coef2'], label='HPI Quartile 2')
ax.plot(df['date'], df['coef3'], label='HPI Quartile 3')
ax.plot(df['date'], df['coef4'], label='HPI Quartile 4')
ax.set_xlabel("Week")
ax.set_ylabel("Distance coefficient")
ax.set_title("Distance coefficient over time")
handles, labels = ax.get_legend_handles_labels() #reverse order of legend
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left') 

plt.xticks(rotation=45) 
plt.tight_layout() 
plt.savefig(f'{outdir}/Demand/overtime/coefplot', dpi=300, bbox_inches='tight')  # Save the figure to a file

print(f"Results saved to {outdir}/Demand/overtime/demest_coefs.csv")