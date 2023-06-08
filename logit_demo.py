
import pyblp
import pandas as pd
import numpy as np

df = pd.read_csv("product_data_mwe.csv", usecols=['market_ids', 'dist', 'prices', 'collegegrad', 'shares'])
df.describe()
# We don't have price in the model, so prices are all zero
# There is one firm per market
# market_ids are ZIP codes
# dist is distance to the nearest pharmacy site
# shares is the vaccination rate, winsorized to [0.05,0.95]


# 1. Logit. This gives us the expected results.
logit_formulation = pyblp.Formulation('1 + prices + collegegrad + log(dist)')
logit_problem = pyblp.Problem(logit_formulation, df)
logit_results = logit_problem.solve()
print(logit_results)


# 2. Include distance as a demographic variable instead.
X1_formulation = pyblp.Formulation('1 + prices + collegegrad')
X2_formulation = pyblp.Formulation('0 + log(dist)')
formulation = (X1_formulation, X2_formulation)
agent_formulation = pyblp.Formulation('0 + log(dist)')

# Make agent data with one agent per market.
agent_data = df[['market_ids', 'dist']]
agent_data = agent_data.assign(nodes = 0, weights = 1)

problem = pyblp.Problem(formulation, df, agent_formulation, agent_data)

iteration_config = pyblp.Iteration(method='squarem', method_options={'atol':1e-12})
optimization_config = pyblp.Optimization('trust-constr', {'gtol':1e-12})

# Estimate coefficient on log(dist). Force variance on random coefficient to zero.
results = problem.solve(pi=-0.1, iteration=iteration_config, optimization=optimization_config, sigma=0)
print(results) # Returns the initial value for pi. 
