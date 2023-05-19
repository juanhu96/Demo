# Test pyblp

import pandas as pd
import pyblp
import numpy as np
import matplotlib.pyplot as plt


product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
product_data.columns

agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
agent_data.columns
agent_data = agent_data[['market_ids', 'city_ids', 'quarter', 'weights', 'nodes0', 'nodes1', 'income', 'income_squared', 'age', 'child']]
agent_data.head()


X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
X2_formulation = pyblp.Formulation('1 + prices')
product_formulations = (X1_formulation, X2_formulation)
product_formulations


agent_formulation = pyblp.Formulation('0 + income')
agent_formulation

nevo_problem = pyblp.Problem(
    product_formulations,
    product_data,
    agent_formulation,
    agent_data
)
nevo_problem


res = nevo_problem.solve(beta = np.zeros((1,1)), sigma=np.ones((2, 2)), pi = np.array([0,pi_init]), optimization=pyblp.Optimization('return'))

res.objective[0][0]



objective_log = []
pi_init_log = np.linspace(-20,-10,11)
for pi_init in pi_init_log:
    res = nevo_problem.solve(beta = np.zeros((1,1)), sigma=np.ones((2, 2)), pi = np.array([0,pi_init]), optimization=pyblp.Optimization('return'))
    objective_log.append(res.objective[0][0])
    print(pi_init, res.objective[0][0])



fig, ax = plt.subplots()
ax.set_xlabel('pi_init')
ax.set_ylabel('objective')
ax.set_title('Objective function')
ax.plot(pi_init_log, objective_log)
plt.savefig(f"/mnt/staff/zhli/objplot_pyblp_{str(int(pi_init_log[0]))}_{str(int(pi_init_log[-1]))}.png")

