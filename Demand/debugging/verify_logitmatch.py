
######## PYBLP ########
import pyblp
import pandas as pd
import numpy as np



datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"
agent_data = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")
df = pd.read_csv(f"{datadir}/Analysis/product_data_tracts.csv")

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
            'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
            'medianhomevalue', 'popdensity', 'population']
formula_str = "1 + prices +  " + " + ".join(controls)
formulation1 = pyblp.Formulation(formula_str + '+ C(hpiquartile)')
formulation2 = pyblp.Formulation('0+log(dist)')
agent_formulation = pyblp.Formulation('0+log(dist)')
pi_init = -0.1

# mwe - one control, log dist, match with logit 
# simple data - minimal code

logit_formulation = pyblp.Formulation(formula_str + '+ log(dist)*C(hpiquartile)')
logit_formulation
logit_problem = pyblp.Problem(logit_formulation, df)
logit_results = logit_problem.solve()
logit_results # verified that logit matched with stata
