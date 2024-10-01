#=================================================================
# TEST -- insert after creating economy object in demest_assm.py
import copy
verbose = True
iter = 0
dists_mm_sorted, sorted_indices, wdists, mm_where= fp.wdist_init(cw_pop, economy.dists)
agent_unique_data = agent_data_full.drop_duplicates(subset=['blkid']).copy()
offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
assert len(offer_weights) == agent_data_full.shape[0]
offer_inds = np.flatnonzero(offer_weights)
offer_weights = offer_weights[offer_inds]
print(f"Number of agents: {len(offer_inds)}")
agent_loc_data = agent_data_full.loc[offer_inds].copy()
agent_loc_data['weights'] = offer_weights/agent_loc_data['population'] #population here is ZIP code population
pi_init = None
results = de.estimate_demand(df, agent_loc_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=1e-8, poolnum=4, verbose=verbose)

coefs = results.pi.flatten()
agent_results = de.compute_abd(results, df, agent_unique_data, coefs=coefs, verbose=verbose)
abd = agent_results['abd'].values
distcoefs = agent_results['distcoef'].values

print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")

a0 = copy.deepcopy(economy.assignments)

af.random_fcfs(economy, distcoefs, abd, capacity, mnl=mnl)

af.assignment_stats(economy, max_rank=len(economy.offers[0]))

# look at which assignments have >0 in the 299th position
blocks_overcap = np.where(np.array(economy.assignments)[:, 299] > 0)[0]
len(blocks_overcap)
agents_overcap_df = agent_results.iloc[blocks_overcap, :]
agents_overcap_df.head(10)
len(agents_overcap_df.market_ids.unique())
# TODO: look at the assignments to see where the violations are

#=================================================================

