import pandas as pd
import numpy as np
import pyblp
try:
    from demand_utils import vax_entities as vaxclass
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
    from demand_utils import fixed_point as fp
except:
    from Demand.demand_utils import vax_entities as vaxclass
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de
    from Demand.demand_utils import fixed_point as fp




datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result/Demand"

setting_tag = "10000_5_4q_mnl"


cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df = de.hpi_dist_terms(df, nsplits=4, add_hpi_bins=True, add_hpi_dummies=True, add_dist=False)
results = pyblp.read_pickle(f"{outdir}/results_{setting_tag}.pkl")

# #=================================================================
###testing###
# read in agent_results
agent_results = pd.read_csv(f"{outdir}/agent_results_{setting_tag}.csv")
agent_results.columns
# median abd 
agent_withpop = agent_results.merge(cw_pop[['blkid', 'population']], on='blkid', how='left')
blockpops_int = np.round(agent_withpop.population.values).astype(int)
abd_expandpop = np.repeat(agent_withpop.abd.values, blockpops_int)
median_abd = np.median(abd_expandpop)
print(f"Median abd: {median_abd}")

distdf_nearest = distdf.groupby('blkid').head(1).reset_index(drop=True)
# 25th/75th percentile of nearest logdist 
logdist_25 = distdf_nearest.logdist.quantile(0.25)
logdist_75 = distdf_nearest.logdist.quantile(0.75)

# median logdist of nearest pharmacy, 5th nearest pharmacy, 10th nearest pharmacy
median_logdist_rank1 = distdf_nearest.logdist.median()
median_logdist_rank5 = distdf.groupby('blkid').head(5).reset_index(drop=True).groupby('blkid').tail(1).reset_index(drop=True).logdist.median()
median_logdist_rank10 = distdf.groupby('blkid').head(10).reset_index(drop=True).groupby('blkid').tail(1).reset_index(drop=True).logdist.median()
print(f"Median dist of nearest pharmacy: {np.exp(median_logdist_rank1)}km.\nMedian dist of 5th nearest pharmacy: {np.exp(median_logdist_rank5)}km.\nMedian dist of 10th nearest pharmacy: {np.exp(median_logdist_rank10)}km.")

coefs = results.pi.flatten()


def share(abd, d, coef):
    u = np.exp(coef * d + abd)
    return u / (1 + u)

# shares at median abd, assuming HPIQ1
print(f"Dist (km): {np.exp(logdist_25)}")
print(f"P25 dist, Q1: {share(median_abd, logdist_25, coefs[0])}")
print(f"Dist (km): {np.exp(logdist_75)}")
print(f"P75 dist, Q1: {share(median_abd, logdist_75, coefs[0])}")

print(f"Dist (km): {np.exp(median_logdist_rank1)}")
print(f"Median nearest dist, Q1: {share(median_abd, median_logdist_rank1, coefs[0])}")
print(f"Dist (km): {np.exp(median_logdist_rank5)}")
print(f"Median 5th nearest dist, Q1: {share(median_abd, median_logdist_rank5, coefs[0])}")
print(f"Dist (km): {np.exp(median_logdist_rank10)}")
print(f"Median 10th nearest dist, Q1: {share(median_abd, median_logdist_rank10, coefs[0])}")

# #=================================================================
# Margins plot (0-10km, 4 HPI quartiles)
# find median abd by hpi quartile
abd_byq = []
for qq in range(1, 5):
    agent_withpop_qq = agent_withpop[agent_withpop.hpi_quantile == qq]
    blockpops_int_qq = np.round(agent_withpop_qq.population.values).astype(int)
    abd_expandpop_qq = np.repeat(agent_withpop_qq.abd.values, blockpops_int_qq)
    median_abd_qq = np.median(abd_expandpop_qq)
    abd_byq.append(median_abd_qq)
print(abd_byq)

distmesh = np.linspace(0, 10, 100)
shares_byq = []
for qq in range(1, 5):
    shares_qq = [share(abd_byq[qq-1], np.log(d), coefs[qq-1]) for d in distmesh]
    shares_byq.append(shares_qq)

## plot without the SE bands
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# colors = ['red', 'orange', 'green', 'blue']
# labels = ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']
# for qq, color, label in zip(range(4), colors, labels):
#     plt.plot(distmesh, np.array(shares_byq[qq]) * 100, label=label, linewidth=2, color=color)

# plt.xlabel("Distance to nearest vaccination site (km)", fontsize=12)
# plt.ylabel("Vaccinated (%)", fontsize=12)

# # Reverse the order of legend entries and move it outside the plot
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles=handles[::-1], labels=labels[::-1], title='Healthy Places Index (quartile)',
#            frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=12)

# plt.tight_layout()
# plt.savefig(f"{outdir}/margins/margins_{setting_tag}.png")


# #=================================================================
# SE bands
# #=================================================================
problem = results.problem
byvar = 'hpi_quantile'
idf = agent_withpop.drop(columns=['hpi_quantile'])
byvals = set(idf[byvar]) # {1, 2, 3, 4}
print("parameters", results.parameters)
Vmat = results.parameter_covariances
dist_coefs = results.pi.flatten()
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df = de.hpi_dist_terms(df, nsplits=4, add_hpi_bins=True, add_hpi_dummies=True, add_dist=False)
df['meanutil'] = results.compute_delta(market_id=df['market_ids'])
df_marg = df.merge(idf, on='market_ids', how='right')
zip_populations = df_marg.groupby('market_ids').population.sum()
df = df.assign(population = df['market_ids'].map(zip_populations))
df_marg.sort_values(by=['market_ids'], inplace=True)
dist_mesh_unlog = np.linspace(0.1, 10, 100)
dist_mesh_log = np.log(dist_mesh_unlog)
df_marg = df_marg.loc[df_marg.index.repeat(len(dist_mesh_log))].reset_index(drop=True) # Expand df_marg (one row per distance)
df_marg['logdist_m'] = np.tile(dist_mesh_log, len(df_marg) // len(dist_mesh_log))
df_marg['dist_util'] = 0
for (qq, qqval) in enumerate(byvals):
    df_marg['dist_util'] += df_marg['logdist_m'] * dist_coefs[qq] * (df_marg[byvar] == qqval)

df_marg['u_i'] = df_marg['meanutil'] + df_marg['dist_util']
df_marg['share_i'] = np.exp(df_marg['u_i']) / (1 + np.exp(df_marg['u_i']))

# populate a vector with the X values
dudb = [np.zeros((Vmat.shape[0], len(dist_mesh_log))) for _ in range(len(byvals))] #4 matrices of 23 x 100 
for (qq,qqval) in enumerate(byvals):
    for (qq2,qqval2) in enumerate(byvals): # 4 nonlinear vars
        dudb[qq][qq2,:] = dist_mesh_log * (qqval == qqval2)
    for (ii,vv) in enumerate(results.beta_labels): # 19 linear vars
        mean_vv = np.average(problem.products[vv].flatten()[df[byvar]==qqval], weights=df.loc[df[byvar]==qqval, 'population']) if vv != '1' else 1
        dudb[qq][ii+qq2+1,:] = mean_vv

pred_s_df = df_marg.groupby([byvar, 'logdist_m']).apply(lambda x: np.average(x['share_i'], weights=x['population'])).reset_index(name='pred_s')
pred_s = pred_s_df.pred_s.values
dsdu = pred_s * (1 - pred_s)
dsdu = dsdu.reshape(1, -1)

dsdb = [np.zeros((Vmat.shape[0], len(dist_mesh_log))) for _ in range(len(byvals))]
Vse = [np.zeros((len(dist_mesh_log),)) for _ in range(len(byvals))]
for (qq,qqval) in enumerate(byvals):
    qqind_in_s = np.where(pred_s_df[byvar] == qqval)[0]
    dsdu_q = dsdu[:,qqind_in_s]
    dsdb[qq] = dsdu_q * dudb[qq]
    Vmarg = dsdb[qq].T @ (Vmat/problem.N) @ dsdb[qq]
    Vse[qq] = np.diag(Vmarg)**0.5

Vse_concat = np.concatenate(Vse, axis=0)
s_ub = pred_s + 1.96 * Vse_concat
s_lb = pred_s - 1.96 * Vse_concat

byvals_rep = np.concatenate([np.repeat(ii, len(dist_mesh_log)) for ii in byvals], axis=0)
df_out = pd.DataFrame({'logdist_m': np.tile(dist_mesh_log, len(byvals)), 'share_i': pred_s, 'share_ub': s_ub, 'share_lb': s_lb, byvar: byvals_rep})
# un-log logdist_m
df_out = df_out.assign(dist_m = np.exp(df_out['logdist_m']))
for share_var in ['share_i', 'share_ub', 'share_lb']:
    df_out[share_var] = df_out[share_var] * 100
savepath = f"{datadir}/Analysis/Demand/marg_{setting_tag}.dta"
df_out.to_stata(savepath, write_index=False)
df_out

# #=================================================================
# #=================================================================







# interquartile range of vax rates
print(df.shares.quantile(0.75) - df.shares.quantile(0.25))
# #=================================================================

# DISTANCE BINS

setting_tag = "10000_300_3q_distbins_at1_5"

agent_results_full = pd.read_csv(f"{outdir}/agent_results_{setting_tag}.csv")
agent_results = agent_results_full[agent_results_full.hpi_quantile == 1]
agent_withpop = agent_results.merge(cw_pop[['blkid', 'population']], on='blkid', how='left')
blockpops_int = np.round(agent_withpop.population.values).astype(int)
abd_expandpop = np.repeat(agent_withpop.abd.values, blockpops_int)
median_abd = np.median(abd_expandpop)
print(f"Median abd: {median_abd}")
results = pyblp.read_pickle(f"{outdir}/results_{setting_tag}.pkl")
coefs = results.pi.flatten()
print(results.pi_labels)


def share_bin(abd, coef):
    u = np.exp(coef + abd)
    return u / (1 + u)



print(f"Distance 0-1km: {share_bin(median_abd, 0):.3f}")
print(f"Distance 1-5km: {share_bin(median_abd, coefs[0]):.3f}")
print(f"Distance 5+ km: {share_bin(median_abd, coefs[1]):.3f}")