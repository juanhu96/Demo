
# START OF MWE
import pyblp
import pandas as pd
import numpy as np

df = pd.read_csv("product_data_mwe.csv", usecols=['market_ids', 'dist', 'prices', 'collegegrad', 'shares'])

# simple logit
logit_formulation = pyblp.Formulation('1 + prices + collegegrad + log(dist)')
logit_problem = pyblp.Problem(logit_formulation, df)
logit_results = logit_problem.solve()
print(logit_results)


# with distance in the agent data
iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12})
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-12})
formulation = (pyblp.Formulation('1 + prices + collegegrad'), pyblp.Formulation('0 + log(dist)'))
agent_formulation = pyblp.Formulation('0 + log(dist)')
agent_data = df[['market_ids', 'dist']]
agent_data = agent_data.assign(nodes = 0, weights = 1)
problem = pyblp.Problem(formulation, df, agent_formulation, agent_data)
results = problem.solve(pi=-0.1, iteration=iteration_config, optimization=optimization_config, sigma=0)
print(results)

df.describe()

# set a column as one
df['aux1'] = 1
formulation = (pyblp.Formulation('1 + prices + collegegrad'), pyblp.Formulation('0 + aux1'))
agent_formulation = pyblp.Formulation('0 + log(dist)')
agent_data = df[['market_ids', 'dist']]
agent_data = agent_data.assign(nodes = 0, weights = 1)
problem2 = pyblp.Problem(formulation, df, agent_formulation, agent_data)
results2 = problem2.solve(pi=-0.1, error_punishment=3, iteration=iteration_config, optimization=optimization_config, sigma=0)
results2
