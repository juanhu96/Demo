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
# distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")
df = de.hpi_dist_terms(df, nsplits=4, add_hpi_bins=True, add_hpi_dummies=True, add_dist=False)
results = pyblp.read_pickle(f"{outdir}/results_{setting_tag}.pkl")

# read in agent_results
agent_results = pd.read_csv(f"{outdir}/agent_results_{setting_tag}.csv")
agent_results.columns
# median abd 
agent_withpop = agent_results.merge(cw_pop[['blkid', 'population']], on='blkid', how='left')

def share(abd, d, coef):
    u = np.exp(coef * d + abd)
    return u / (1 + u)

# #=================================================================
# Margins Plot by HPI Quantile with SE bands
# #=================================================================
problem = results.problem
byvar = 'hpi_quantile'
byvals = set(agent_withpop[byvar]) # {1, 2, 3, 4}
idf = agent_withpop.drop(columns=['hpi_quantile'])
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

# predicted shares (point estimates)
pred_s_df = df_marg.groupby([byvar, 'logdist_m']).apply(lambda x: np.average(x['share_i'], weights=x['population'])).reset_index(name='pred_s')
pred_s = pred_s_df.pred_s.values

# SE bands
for (qq, qqval) in enumerate(byvals):
    for (dd, ddval) in enumerate(dist_mesh_log):
        print(f"Computing SE for HPI Quantile {qqval} and distance {ddval}")
        df_marg_qd = df_marg[(df_marg[byvar] == qqval) & (df_marg['logdist_m'] == ddval)]
        dudb_qd = np.zeros((len(results.beta_labels)+len(results.pi_labels), df_marg_qd.shape[0]))
        dudb_qd[qq,:] = np.tile(dist_mesh_log, df_marg_qd.shape[0] // len(dist_mesh_log))
        for (ii,vv) in enumerate(results.beta_labels): # 19 linear vars
            dudb_qd[ii+len(byvals),:] = df_marg_qd[vv] if vv != '1' else 1
        # weights_qq = np.array(df_marg_qd['population']).reshape(-1, 1)
        weights_qq = np.array(df_marg_qd['population']).reshape(1, -1)
        weights_qq.shape
        dsdu_qq = df_marg_qd['share_i'] * (1 - df_marg_qd['share_i'])
        dsdu_qq = np.array(dsdu_qq).reshape(1, -1)
        dsdb_qq = dsdu_qq * dudb_qd
        dsdb_qq.shape
        qq_pop = np.sum(df_marg_qd['population'])
        Vmarg_qq = dsdb_qq.T @ (Vmat/qq_pop) @ dsdb_qq
        break 
    break









# populate a vector with the X values
dudb = [np.zeros((Vmat.shape[0], len(dist_mesh_log))) for _ in range(len(byvals))] #4 matrices of 23 x 100 
for (qq,qqval) in enumerate(byvals):
    for (qq2,qqval2) in enumerate(byvals): # 4 nonlinear vars
        dudb[qq][qq2,:] = dist_mesh_log * (qqval == qqval2)
    for (ii,vv) in enumerate(results.beta_labels): # 19 linear vars
        mean_vv = np.average(problem.products[vv].flatten()[df[byvar]==qqval], weights=df.loc[df[byvar]==qqval, 'population']) if vv != '1' else 1
        dudb[qq][ii+qq2+1,:] = mean_vv

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
# Share difference between 1km and 2km - integrating over the distribution of ABD
# #=================================================================

# difference in share between 1km->2km for HPIQ1
share_q1_1km = df_out.loc[(df_out.hpi_quantile == 1) & (df_out.dist_m == 1), 'share_i'].values[0]
share_q1_2km = df_out.loc[(df_out.hpi_quantile == 1) & (df_out.dist_m == 2), 'share_i'].values[0]
diff_q1_1to2km = share_q1_2km - share_q1_1km
print(f"Difference in share between 1km->2km for HPIQ1: {diff_q1_1to2km:.3f}")
# difference in share between 1km->2km for HPIQ4
share_q4_1km = df_out.loc[(df_out.hpi_quantile == 4) & (df_out.dist_m == 1), 'share_i'].values[0]
share_q4_2km = df_out.loc[(df_out.hpi_quantile == 4) & (df_out.dist_m == 2), 'share_i'].values[0]
diff_q4_1to2km = share_q4_2km - share_q4_1km
print(f"Difference in share between 1km->2km for HPIQ4: {diff_q4_1to2km:.3f}")



