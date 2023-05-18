import pandas as pd
import pyblp
import numpy as np

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"


# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")

# Formulations
formulation1 = pyblp.Formulation('1 + prices')
formulation2 = pyblp.Formulation('0 + log(dist)')
product_formulation = (formulation1, formulation2)
agent_formulation = pyblp.Formulation('1')


# Create agent data
create_agent_data = False #Switch. If True, create agent_data.csv. If False, read it in (for consistency).
if create_agent_data:
    agent_data = pd.DataFrame()
    ndraws = 50
    agent_data['market_ids'] = np.tile(df['market_ids'].unique(), ndraws)
    agent_data['nodes0'] = np.random.normal(size=(len(agent_data),1))
    agent_data['nodes1'] = np.random.normal(size=(len(agent_data),1))
    agent_data['weights'] = 1 / ndraws
    agent_data.to_csv(f"{datadir}/Analysis/agent_data.csv", index = False)
else:
    agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")


problem = pyblp.Problem(
    product_formulations=product_formulation, product_data=df,
    agent_formulation=agent_formulation, agent_data=agent_data,)
print(problem)



####################
####################
####################


res = problem.solve(beta = np.zeros((2,1)), sigma = 0, pi = -0.5) 

np.corrcoef(df.logdist, df.shares)