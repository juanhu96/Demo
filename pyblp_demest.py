import pandas as pd
import pyblp
import numpy as np
poolnum = 24
pyblp.options.digits = 4

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0
df.hpiquartile.unique()
cat_dtype = pd.CategoricalDtype(categories=[4,3,2,1], ordered=True) #TODO: figure out how to make 4 the reference category
df.hpiquartile = df.hpiquartile.astype(cat_dtype)
print(df['hpiquartile'])


formulation1 = pyblp.Formulation("1 + prices + log(dist)/C(hpiquartile) + race_black + race_asian + race_hispanic + race_other + health_employer + health_medicare + health_medicaid + health_other + collegegrad + unemployment + poverty + medianhhincome + medianhomevalue + popdensity + population", absorb='C(hpiquartile)')

include_hpi = False
if include_hpi:
    formulation2 = pyblp.Formulation('0 + log(dist):C(hpiquartile)')
    sigma_init = 0.1*np.eye(4)
    picklepath = f"{datadir}/Analysis/res_byhpi.pkl"
    dfpath = f"{datadir}/Analysis/demest_byhpi.csv"
else:
    formulation2 = pyblp.Formulation('0 + log(dist)')
    sigma_init = 0.1
    picklepath = f"{datadir}/Analysis/res_pyblp.pkl"
    dfpath = f"{datadir}/Analysis/demest_pyblp.csv"

product_formulation = (formulation1, formulation2)


integ_config = pyblp.Integration('monte_carlo', size=100, specification_options={'seed': 0})
opt_config = pyblp.Optimization('trust-constr', {'gtol': 1e-10})

problem = pyblp.Problem(product_formulations=product_formulation, product_data=df, integration=integ_config)

print(problem)

####################
#################### solve


re_estimate = True #Switch to False to read the pickle
if re_estimate:
    with pyblp.parallel(poolnum):
        res = problem.solve(sigma = sigma_init, optimization=opt_config)
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

# save results
df.to_csv(dfpath)
print("saved results at: ", dfpath)


####################
####################
####################
# plot objective function for different distance coefficients
plot_obj = False
if plot_obj:
    import matplotlib.pyplot as plt
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