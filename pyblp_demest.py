import pandas as pd
import pyblp
import numpy as np
poolnum = 32
pyblp.options.digits = 4

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# Load data
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df['prices'] = 0
df.hpiquartile.unique()
# cat_dtype = pd.CategoricalDtype(categories=[4,3,2,1], ordered=True) #TODO: figure out how to make 4 the reference category
# df.hpiquartile = df.hpiquartile.astype(cat_dtype)
print(df['hpiquartile'])

controls = ['race_black', 'race_asian', 'race_hispanic', 'race_other',
            'health_employer', 'health_medicare', 'health_medicaid', 'health_other',
            'collegegrad', 'unemployment', 'poverty', 'medianhhincome', 
            'medianhomevalue', 'popdensity', 'population']
formula_str = "1 + prices + log(dist)/C(hpiquartile) + " + " + ".join(controls)
formulation1 = pyblp.Formulation(formula_str, absorb='C(hpiquartile)')

include_hpi = False #Switch to True to include HPI quartile * distance interaction in the RC
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


re_estimate = False #Switch to False to read the pickle
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

len(res.beta_labels)
res.beta

# Margins plots
# deltas = res.compute_delta()
# df['shares_fromdelta'] = np.exp(deltas) / (1 + np.exp(deltas))
# np.corrcoef(df['shares_fromdelta'], df['shares'])

# mean product
deltas = res.compute_delta().flatten()
df['xi_fe'] = res.xi_fe
df['xi'] = res.xi
df.groupby('hpiquartile')[['xi_fe']].describe()

v = res.problem.agents.nodes.flatten()
agent_mktids = res.problem.agents.market_ids.flatten()
rc = res.sigma[0,0]*v

idf = pd.DataFrame({'rc':rc, 'market_ids':agent_mktids})
idf

distbetas = [res.beta[1], res.beta[1] + res.beta[2], res.beta[1] + res.beta[3], res.beta[1] + res.beta[4]]
distbetas = np.array(distbetas).flatten()

controlbetas = res.beta[5:].flatten()
len(controlbetas)
df['meanutil'] = np.dot(df[controls], controlbetas)
# create hpiquartile-specific distance betas
df['distbeta'] = 0
for ii in range(1,5):
    df.loc[df['hpiquartile'] == ii, 'distbeta'] = distbetas[ii-1]

# df['meanutil'] = df['meanutil'] + (np.log(df['dist']) * df['distbeta'])
# mean utility does not contain distance terms right now
df['meanutil'] = df['meanutil'] + df['xi_fe']  + df['xi']

np.corrcoef(df['meanutil'], deltas)
pd.Series(deltas).describe()
pd.Series(df['meanutil']).describe()

# merge df and idf on market_ids
df_marg = df.merge(idf, on='market_ids', how='right')
df_marg.drop(columns=['market_ids'], inplace=True)
df_marg.to_stata(f"{datadir}/Analysis/agents_marg.dta")


list(df_marg)



meandf = df.groupby('hpiquartile')[controls + ['market_ids', 'firm_ids', 'prices', 'xi_fe']].mean().reset_index()
meandf['meanutil'] = np.dot(meandf[controls], res.beta[5:])
meandf['meanutil'] = meandf['meanutil'] + meandf['xi_fe']
meandf['distbeta'] = distbetas
dflist = []
for dd in np.linspace(0.25,10,40):
    dfdd = meandf.assign(dist = dd)
    dflist.append(dfdd)

df_marg = pd.concat(dflist)
# df_marg['meanutil'] = np.dot(df_marg[controls], res.beta[5:])
# df_marg['meanutil'] = df_marg['meanutil'] + df_marg['xi_fe']
df_marg['meanutil'] = df_marg['meanutil'] + np.log(df_marg['dist'])*df_marg['distbeta']
df_marg['shares'] = np.exp(df_marg['meanutil']) / (1 + np.exp(df_marg['meanutil']))
df_marg.to_csv(f"{datadir}/Analysis/demest_margins.csv")



# formulation_noabsorb = (pyblp.Formulation(formula_str), formulation2)
# formulation1
# beta_withconst = np.concatenate(([[0]],res.beta))
# res.beta_labels
# sim = pyblp.Simulation(product_formulations=formulation_noabsorb, product_data=df_marg, beta = beta_withconst, sigma = res.sigma, xi = df_marg.xi_fe, integration=integ_config)



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