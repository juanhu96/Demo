import pandas as pd
import pyblp
import numpy as np
import matplotlib.pyplot as plt

poolnum = 24

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0

# Formulations
formulation1 = pyblp.Formulation('1 + prices + log(dist) + race_black + race_asian + race_hispanic + race_other + health_employer + health_medicare + health_medicaid + health_other + collegegrad + unemployment + poverty + medianhhincome + medianhomevalue + popdensity + population')
formulation2 = pyblp.Formulation('0 + log(dist)')
product_formulation = (formulation1, formulation2)

integ_config = pyblp.Integration('monte_carlo', size=100, specification_options={'seed': 0})
opt_config = pyblp.Optimization('trust-constr', {'gtol': 1e-10})

problem = pyblp.Problem(product_formulations=product_formulation, product_data=df, integration=integ_config)

print(problem)

####################
####################
#################### solve it


re_estimate = True #Switch to False to read the pickle
picklepath = f"{datadir}/Analysis/res_pyblp.pkl"
if re_estimate:
    with pyblp.parallel(poolnum):
        res = problem.solve(sigma = 0.1, optimization=opt_config)
    res.to_pickle(picklepath)
else:
    res = pyblp.read_pickle(picklepath)

print(res)
print('\nBeta:')
for ii in range(len(res.beta)):
    print(res.beta_labels[ii], res.beta[ii])
print('\nSigma:')
for ii in range(len(res.sigma)):
    print(res.sigma_labels[ii], res.sigma[ii])


# summary stats
de = res.compute_elasticities('dist')
df['dist_elast'] = de
print("Elasticities:")
print(df['dist_elast'].describe())
print(df.groupby('hpiquartile')['dist_elast'].describe())


####################
####################
##################### 
# plot objective function for different distance coefficients
plot_obj = False
if plot_obj:
    pyblp.options.verbose = False

    objective_log = []
    dist_coefs = np.linspace(-4,2,25)
    for bb in dist_coefs:
        beta_init = res.beta
        beta_init[res.beta_labels.index('log(dist)')] = bb
        with pyblp.parallel(poolnum):
            res = problem.solve(beta = beta_init, sigma = res.sigma, optimization=pyblp.Optimization('return'))
        objective_log.append(res.objective[0][0])
        print(bb, res.objective[0][0])

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('distance coefficient')
    ax.set_ylabel('objective')
    ax.set_title('Objective function')
    ax.plot(dist_coefs, objective_log)
    figpath = f"/mnt/staff/zhli/objplot_pyblp_{str(int(dist_coefs[0]))}_{str(int(dist_coefs[-1]))}.png"
    print("saving figure at: ", figpath)
    plt.savefig(figpath)