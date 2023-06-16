# Margins after demest_tracts_pooled.py

import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3

datadir = "/export/storage_adgandhi/MiscLi/VaccineDemandLiGandhi/Data"


config = [False, False, False]


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
if byvar == 'pooled':
    idf = idf.assign(pooled = 1)

results = pyblp.read_pickle(f"{datadir}/Analysis/tracts_results_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.pkl")

Vmat = results.parameter_covariances

dist_coefs = results.pi.flatten()
dist_coefs_se = results.pi_se.flatten()

idf = idf.assign(distbeta = 0, distbeta_se = 0)
for qq in set(idf[byvar]):
    qq = int(qq)
    is_q = idf[byvar] == qq
    idf['distbeta'] += is_q * dist_coefs[qq-1] 
    idf['distbeta_se'] += is_q * dist_coefs_se[qq-1]
    
# mean utility (zip-level)
df['meanutil'] = results.compute_delta(market_id = df['market_ids'])

df_marg = df.merge(idf, on='market_ids', how='right')
df_marg['weights'] = df_marg['population'] * df_marg['weights']

df_marg.sort_values(by=['market_ids'], inplace=True)



df_marg['agent_id'] = np.arange(len(df_marg))

# Expand DataFrame and create logdist_m
dist_mesh = np.linspace(-0.5, 2.5, 31)
df_marg = df_marg.loc[df_marg.index.repeat(len(dist_mesh))].reset_index(drop=True)
df_marg['logdist_m'] = np.tile(dist_mesh, len(df_marg) // len(dist_mesh))

df_marg['u_i'] = df_marg['meanutil'] + df_marg['distbeta'] * df_marg['logdist_m']
df_marg['share_i'] = np.exp(df_marg['u_i']) / (1 + np.exp(df_marg['u_i']))
# df_marg['dsdu'] = df_marg['share_i'] * (1 - df_marg['share_i'])

# # mean dsdu by logdist_m
# dsdu = df_marg.groupby(['logdist_m'])['dsdu'].mean()


# populate a vector with the X values
dudb = np.zeros((Vmat.shape[0], len(dist_mesh)))
dudb.shape

results.pi_labels
results.beta_labels
dudb[0, :] = dist_mesh
for (ii,vv) in enumerate(results.beta_labels):
    print(vv)
    # TODO: if it's by quartile, this should be the mean within quartile
    mean_vv = np.average(df[vv], weights=df['population']) if vv != '1' else 1
    dudb[ii+1, :] = mean_vv
    

pred_s = df_marg.groupby('logdist_m').apply(lambda x: np.average(x['share_i'], weights=x['weights']))
dsdu = pred_s * (1 - pred_s)

dsdu = dsdu.values.reshape(1, -1)
# Multiply each row of dudb with dsdu
dsdb = dudb * dsdu

Vmarg = (dsdb.T @ (Vmat/results.problem.N) @ dsdb)
Vse = np.diag(Vmarg)**0.5

s_ub = pred_s + 1.96 * Vse
s_lb = pred_s - 1.96 * Vse

s_ub

df_out = pd.DataFrame({'logdist_m': dist_mesh, 'share_i': pred_s, 'share_ub': s_ub, 'share_lb': s_lb})



# pi_se = results.pi_se[0][0]
# v_mat = results.parameter_covariances 
# (v_mat[0,0]/results.problem.N)**0.5






df_out.to_stata(f"{datadir}/Analysis/tracts_marg_pooled_{int(include_hpiquartile)}{int(interact_disthpi)}{int(include_controls)}.dta", write_index=False)




