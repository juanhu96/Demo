import pandas as pd
import pyblp
import numpy as np
import matplotlib.pyplot as plt
import sys

poolnum = 24

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0
# TODO: how to construct the distXhpi variables? Set to 0 -> divide by 0. Missing -> error. Set to 1 -> log to zero -> probably wrong elasticities. 
# Possibly: try with linear/quadratic distance (is that correct?)? Try with demographics?

#Controls: 
# + race_black + race_asian + race_hispanic + race_other + health_employer + health_medicare + health_medicaid + health_other + collegegrad + unemployment + poverty + medianhhincome + medianhomevalue + popdensity + population

include_hpi = True
recompute_log = False #whether to re-compute log(distXhpi) or use the pre-computed logdistXhpi. distXhpi=1 if the ZIP is not in the HPI quartile. 

if include_hpi:
    sigma_init = 0.1*np.eye(4)
else:
    sigma_init = 0.1*np.eye(1)

if recompute_log:
    formulation1 = pyblp.Formulation('1 + prices + log(distXhpi1) + log(distXhpi2) + log(distXhpi3) + log(distXhpi4)', absorb='C(hpiquartile)')
    if include_hpi:
        formulation2 = pyblp.Formulation('0 + log(distXhpi1) + log(distXhpi2) + log(distXhpi3) + log(distXhpi4)')
    else:
        formulation2 = pyblp.Formulation('0 + log(dist)')
else:
    formulation1 = pyblp.Formulation('1 + prices + logdistXhpi1 + logdistXhpi2 + logdistXhpi3 + logdistXhpi4', absorb='C(hpiquartile)')
    if include_hpi:
        formulation2 = pyblp.Formulation('0 + logdistXhpi1 + logdistXhpi2 + logdistXhpi3 + logdistXhpi4')
    else:
        formulation2 = pyblp.Formulation('0 + logdist')

re_interact = True #whether to re-interact the distXhpi variables via the formulation.
if re_interact:
    formulation1 = pyblp.Formulation('1 + prices + log(dist):C(hpiquartile)', absorb='C(hpiquartile)')
    formulation2 = pyblp.Formulation('0 + log(dist):C(hpiquartile)')

product_formulation = (formulation1, formulation2)

# integration configuation
integ_config = pyblp.Integration('monte_carlo', size=50, specification_options={'seed': 0})
# integ_config = pyblp.Integration('product', size = 5, specification_options={'seed': 0})

# optimization configuration
opt_config = pyblp.Optimization('trust-constr', {'gtol': 1e-6})

problem = pyblp.Problem(product_formulations=product_formulation, product_data=df, integration=integ_config)

print(problem)

type(formulation2)


####################
####################
#################### solve it


re_estimate = True #Switch to False to read the pickle

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
for vv in ['distXhpi1', 'distXhpi2', 'distXhpi3', 'distXhpi4']:
    de = res.compute_elasticities(vv)
    print(f"\n\nVariable: {vv}")
    print(df[vv].describe())
    print("Elasticities:")
    print(pd.Series(de.flatten()).describe())
    


####################
####################
##################### 
# plot objective function for different pi_init
plot_obj = False
if plot_obj:
    pyblp.options.verbose = False

    objective_log = []
    dist_coefs = np.linspace(-5,2,15)
    for bb in dist_coefs:
        beta_init = res.beta
        beta_init[res.beta_labels.index('log(dist)')] = bb
        with pyblp.parallel(poolnum):
            res = problem.solve(beta = beta_init, sigma = res.sigma, optimization=pyblp.Optimization('return'))
        objective_log.append(res.objective[0][0])
        print(bb, res.objective[0][0])
        sys.stdout.flush()

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('pi_init')
    ax.set_ylabel('objective')
    ax.set_title('Objective function')
    ax.plot(dist_coefs, objective_log)
    figpath = f"/mnt/staff/zhli/objplot_byhpi_{str(int(dist_coefs[0]))}_{str(int(dist_coefs[-1]))}.png"
    print("saving figure at: ", figpath)
    plt.savefig(figpath)