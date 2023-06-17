# Margins after demest_tracts_pooled.py

import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"

# config = [False, False, False]
# config = [True, False, False]
# config = [True, True, False]
config = [True, True, True]


# for config in [
#     [False, False, False],
#     [True, False, False],
#     [True, True, False],
#     [True, True, True]
#     ]:

include_hpiquartile, interact_disthpi, include_controls = config

if interact_disthpi:
    byvar = 'hpi_quartile'
else:
    byvar = 'pooled'

idf = pd.read_csv(f"{datadir}/Analysis/agent_data.csv")
df = pd.read_csv(f"{datadir}/Analysis/demest_data.csv")
df.rename(columns={'hpiquartile': 'hpi_quartile'}, inplace=True) #for consistency with the other quartile-based variables
if byvar == 'pooled':
    idf = idf.assign(pooled = 1)

byvals = set(idf[byvar])
results = pyblp.read_pickle(f"{datadir}/Analysis/tracts_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")
problem = results.problem

print("parameters", results.parameters)
Vmat = results.parameter_covariances
Vmat.shape

dist_coefs = results.pi.flatten()
dist_coefs_se = results.pi_se.flatten()

# mean utility (zip-level, everything but distance)
df['meanutil'] = results.compute_delta(market_id = df['market_ids'])

# df[['meanutil', byvar]].groupby([byvar]).mean()

df = df.rename(columns={byvar: byvar+'_zip'})
idf = idf.rename(columns={byvar: byvar+'_tract'})
df_marg = df.merge(idf, on='market_ids', how='right')
# Weight = ZIP population * agent weights
df_marg['weights'] = df_marg['population'] * df_marg['weights']

df_marg.sort_values(by=['market_ids'], inplace=True)

# Expand df_marg and create logdist_m (distance mesh)
dist_mesh = np.linspace(-0.5, 2.5, 31)
df_marg = df_marg.loc[df_marg.index.repeat(len(dist_mesh))].reset_index(drop=True)
df_marg['logdist_m'] = np.tile(dist_mesh, len(df_marg) // len(dist_mesh))

df_marg = df_marg.assign(dist_util = 0)
for (qq, qqval) in enumerate(byvals):
    df_marg['dist_util'] += df_marg['logdist_m'] * dist_coefs[qq] * (df_marg[byvar+"_tract"] == qqval)


df_marg['u_i'] = df_marg['meanutil'] + df_marg['dist_util']
df_marg['share_i'] = np.exp(df_marg['u_i']) / (1 + np.exp(df_marg['u_i']))

# populate a vector with the X values
dudb = [np.zeros((Vmat.shape[0], len(dist_mesh))) for _ in range(len(byvals))]
dudb[0]
dudb[1]
dudb[2]
dudb[3]
dudb[3].shape
results.parameters
results
for (qq,qqval) in enumerate(byvals):
    for (qq2,qqval2) in enumerate(byvals):
        dudb[qq][qq2,:] = dist_mesh * (qqval == qqval2)
    for (ii,vv) in enumerate(results.beta_labels):
        mean_vv = np.average(problem.products[vv].flatten()[df[byvar+'_zip']==qqval], weights=df.loc[df[byvar+'_zip']==qqval, 'population']) if vv != '1' else 1
        dudb[qq][ii+qq2+1,:] = mean_vv


pred_s_df = df_marg.groupby([byvar+"_zip", 'logdist_m']).apply(lambda x: np.average(x['share_i'], weights=x['weights'])).reset_index(name='pred_s')
pred_s = pred_s_df.pred_s.values
dsdu = pred_s * (1 - pred_s)
dsdu = dsdu.reshape(1, -1)

dsdb = [np.zeros((Vmat.shape[0], len(dist_mesh))) for _ in range(len(byvals))]
Vse = [np.zeros((len(dist_mesh),)) for _ in range(len(byvals))]
for (qq,qqval) in enumerate(byvals):
    qqind_in_s = np.where(pred_s_df[byvar+'_zip'] == qqval)[0]
    dsdu_q = dsdu[:,qqind_in_s]
    dsdb[qq] = dsdu_q * dudb[qq]
    Vmarg = dsdb[qq].T @ (Vmat/problem.N) @ dsdb[qq]
    Vse[qq] = np.diag(Vmarg)**0.5

Vse[0]
dudb[qq].shape
dsdu_q.shape
dsdb[qq].shape
Vmat.shape
Vse[0].shape
Vmat.shape

Vse_concat = np.concatenate(Vse, axis=0)

s_ub = pred_s + 1.96 * Vse_concat
s_lb = pred_s - 1.96 * Vse_concat

if byvar == 'pooled':
    df_out = pd.DataFrame({'logdist_m': dist_mesh, 'share_i': pred_s, 'share_ub': s_ub, 'share_lb': s_lb})
else:
    byvals_rep = np.concatenate([np.repeat(ii, len(dist_mesh)) for ii in byvals], axis=0)
    df_out = pd.DataFrame({'logdist_m': np.tile(dist_mesh, len(byvals)), 'share_i': pred_s, 'share_ub': s_ub, 'share_lb': s_lb, byvar: byvals_rep})

# pi_se = results.pi_se[0][0]
# v_mat = results.parameter_covariances 
# (v_mat[0,0]/problem.N)**0.5


df_out.to_stata(f"{datadir}/Analysis/tracts_marg_pooled_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.dta", write_index=False)



