import pandas as pd
import pyblp
import numpy as np
poolnum = 24
pyblp.options.digits = 4

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0

formulation1 = pyblp.Formulation('1 + prices + log(dist):C(hpiquartile) + race_black + race_asian + race_hispanic + race_other + health_employer + health_medicare + health_medicaid + health_other + collegegrad + unemployment + poverty + medianhhincome + medianhomevalue + popdensity + population', absorb='C(hpiquartile)')

include_hpi = True

if include_hpi:
    formulation2 = pyblp.Formulation('0 + log(dist):C(hpiquartile)')
    sigma_init = 0.1*np.eye(4)
else:
    formulation2 = pyblp.Formulation('0 + log(dist)')
    sigma_init = 0.1


product_formulation = (formulation1, formulation2)

# integration configuation
integ_config = pyblp.Integration('monte_carlo', size=100, specification_options={'seed': 0})
# integ_config = pyblp.Integration('product', size = 5, specification_options={'seed': 0})

# optimization configuration
opt_config = pyblp.Optimization('trust-constr', {'gtol': 1e-10})

problem = pyblp.Problem(product_formulations=product_formulation, product_data=df, integration=integ_config)

print(problem)

####################
####################
#################### solve it


re_estimate = False #Switch to False to read the pickle

picklepath = f"{datadir}/Analysis/res_byhpi.pkl"
if re_estimate:
    with pyblp.parallel(poolnum):
        res = problem.solve(sigma = sigma_init, optimization=opt_config)
    res.to_pickle(picklepath)
else:
    res = pyblp.read_pickle(picklepath)

print(res)
print('\nbeta:')
for ii in range(len(res.beta)):
    print(res.beta_labels[ii], res.beta[ii])
print('\nsigma:')
for ii in range(len(res.sigma)):
    print(res.sigma_labels[ii], res.sigma[ii])


# summary stats
de = res.compute_elasticities('dist')
df['dist_elast'] = de
print("Elasticities:")
print(df['dist_elast'].describe())
print(df.groupby('hpiquartile')['dist_elast'].describe())

