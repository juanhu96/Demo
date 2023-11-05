import pandas as pd
import numpy as np
import pyblp


datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result/Demand"
cw_pop = pd.read_csv(f"{datadir}/Analysis/Demand/block_data.csv", usecols=["blkid", "market_ids", "population"])
distdf = pd.read_csv(f"{datadir}/Intermediate/ca_blk_pharm_dist.csv", dtype={'locid': int, 'blkid': int})
df = pd.read_csv(f"{datadir}/Analysis/Demand/demest_data.csv")

# #=================================================================
###testing###
# read in agent_results
setting_tag = "10000_200_3q"
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

results = pyblp.read_pickle(f"{outdir}/results_{setting_tag}.pkl")
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


# interquartile range of vax rates
print(df.shares.quantile(0.75) - df.shares.quantile(0.25))
# #=================================================================
