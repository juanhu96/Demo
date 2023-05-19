import pandas as pd
import pyblp
import numpy as np
import matplotlib.pyplot as plt
import sys

pyblp.options.verbose = False
poolnum = 24

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"


# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0

# Formulations
# formulation1 = pyblp.Formulation('1 + prices')
formulation1 = pyblp.Formulation('1 + prices + log(dist)')
formulation2 = pyblp.Formulation('0 + log(dist)')
product_formulation = (formulation1, formulation2)
# agent_formulation = pyblp.Formulation('1')


# # Create agent data
# create_agent_data = False #Switch. If True, create agent_data.csv. If False, read it in (for consistency).
# if create_agent_data:
#     agent_data = pd.DataFrame()
#     ndraws = 50
#     agent_data['market_ids'] = np.tile(df['market_ids'].unique(), ndraws)
#     agent_data['nodes0'] = np.random.normal(size=(len(agent_data),1))
#     agent_data['weights'] = 1 / ndraws
#     agent_data.to_csv(f"{datadir}/Analysis/agent_data.csv", index = False)
# else:
#     agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")

# # make trivial agent data
# agent_data = pd.DataFrame()
# agent_data['market_ids'] = np.tile(df['market_ids'].unique(), 1)
# agent_data['nodes0'] = np.zeros((len(agent_data),1))
# agent_data['weights'] = 1


# problem = pyblp.Problem(
#     product_formulations=product_formulation, product_data=df,
#     agent_formulation=agent_formulation, agent_data=agent_data)

problem = pyblp.Problem(product_formulations=product_formulation, product_data=df, integration=pyblp.Integration('monte_carlo', size=100))

print(problem)



####################
####################
##################### plot objective function for different pi_init
pyblp.options.verbose = False

objective_log = []
pi_init_log = np.linspace(-2,1,16)
for pi_init in pi_init_log:
    with pyblp.parallel(poolnum):
        res = problem.solve(beta = np.array([1,0]), sigma = 0, pi = pi_init, optimization=pyblp.Optimization('return'))
    objective_log.append(res.objective[0][0])

    print(pi_init, res.objective[0][0])
    sys.stdout.flush()

fig, ax = plt.subplots()
ax.set_xlabel('pi_init')
ax.set_ylabel('objective')
ax.set_title('Objective function')
ax.plot(pi_init_log, objective_log)
plt.savefig(f"/mnt/staff/zhli/objplot_pyblp_{str(int(pi_init_log[0]))}_{str(int(pi_init_log[-1]))}.png")
####################
####################
#################### solve it
pyblp.options.verbose = True

# res = problem.solve(beta = np.zeros((2,1)), sigma = 0.1, pi = -3, optimization=pyblp.Optimization('trust-constr'))
# res = problem.solve(beta = np.array([np.nan, 0]), sigma = 0.1, pi = -1, optimization=pyblp.Optimization('trust-constr'))
with pyblp.parallel(poolnum):
    res = problem.solve(beta = np.array([0.1, 0, -1]), sigma = 0.1, optimization=pyblp.Optimization('trust-constr'))
res.sigma
res.objective
# no demographics - add distance to the linear part
# check if we have dist in X1 in the PE paper