try:
    from demand_utils.vax_entities import Economy, Individual
except:
    from Demand.demand_utils.vax_entities import Economy, Individual

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r, logistic
from typing import List, Optional, Tuple
import time
from multiprocessing import Pool



def initialize_economy(
    locs: List[np.ndarray], #pharmacy IDs in each geog (ordered)
    dists: List[np.ndarray], #sorted within each geog
    n_individuals, # number of individuals in each geog
    shuffle: bool = True,
    seed: int = 1234,
    epsilon_opt = "logistic" # "logistic" or "gumbel" or "zero" 
    ):

    time1 = time.time()
    print("\nInitializing economy...")
    np.random.seed(seed)
    n_geogs = len(locs)

    # Generate random epsilon_diff = epsilon1 - epsilon0
    if epsilon_opt == "gumbel":
        epsilon_0 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
        epsilon_1 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
        epsilon_diff = [epsilon_1[tt] - epsilon_0[tt] for tt in range(n_geogs)]
    elif epsilon_opt == "logistic":
        epsilon_diff = [logistic.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
    elif epsilon_opt == "zero":
        epsilon_diff = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]

    # "logistic" is equivalent to "gumbel", "zero" is not what we want

    # sort epsilon_diff within each geog, ok since we'll shuffle for assignment
    epsilon_diff = [np.sort(epsilon_diff[tt]) for tt in range(n_geogs)]

    print("Finished generating epsilons:", round(time.time() - time1, 2))


    economy = Economy(
        locs=locs,
        dists=dists,
        individuals = [
            [Individual(epsilon_diff[tt][ii]) for ii in range(n_individuals[tt])]
            for tt in range(n_geogs)
        ],
        shuffle=shuffle
    )

    print("Finished initializing economy:", round(time.time() - time1, 2))
    
    return economy



# condition : ii.u_ij > ii.epsilon0
# condition : ii.epsilon1 + ab_epsilon[ll] > ii.epsilon0
# condition : ii.epsilon0 - ii.epsilon1 < ab_epsilon[ll]
# condition : -epsilon_diff < ab_epsilon[ll]
# condition : epsilon_diff > -ab_epsilon[ll]

# ab_epsilon is sorted descending within each geog since distances are sorted ascending
# epsilon_diff is sorted ascending within each geog

def compute_geog_pref(args: Tuple[List[Individual], np.ndarray]):
    individuals, ab_epsilon = args
    abe_neg = -ab_epsilon
    start_index = 0
    for ii in individuals:
        start_index = np.searchsorted(abe_neg[start_index:], ii.epsilon_diff) + start_index
        ii.nlocs_considered = start_index
    return individuals


def compute_pref(
    economy: Economy, 
    abd, 
    distcoefs, 
    poolnum=32,
    scale = 1
    ):
    
    """
    Compute the consideration set for each individual.
    """
    time1 = time.time()
    economy.abe = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)] 
    if scale != 1:
        economy.abe = [scale * economy.abe[tt] for tt in range(economy.n_geogs)]

    print("Time to compute abe:", round(time.time() - time1, 2))
    print("\nComputing preferences with {} parallel processes...".format(poolnum))
    with Pool(poolnum) as p:
        economy.individuals = p.map(compute_geog_pref, [(economy.individuals[tt], economy.abe[tt]) for tt in range(economy.n_geogs)])

    print("Total time in compute_pref():", round(time.time() - time1, 2), "seconds")

def pref_stats(economy: Economy):
    v_nlocs = np.array([ii.nlocs_considered for tt in economy.individuals for ii in tt])
    print("Fraction considering at least one location:", np.mean(v_nlocs!=0)) # should match to demand estimation with nearest location


def add1_at(lst, pos):
    """
    Add 1 to the element at position pos in the list lst, appending with zeros if necessary.
    """
    if pos < len(lst):
        lst[pos] += 1
    else:
        lst.extend([0] * (pos - len(lst)))
        lst.append(1)
    return lst



def random_fcfs(economy: Economy, 
                capacity: int
                ):
    """
    Assign individuals to locations in random order, first-come-first-serve.
    """
    
    time1 = time.time()
    print("\nAssigning individuals...")

    # reset occupancies 
    economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
    full_locations = set() #set for performance reasons

# TODO: why is not everyone offered something?
    economy.offers = [[] for tt in range(economy.n_geogs)]
    economy.assignments = [[] for tt in range(economy.n_geogs)]
    offers = economy.offers
    assignments = economy.assignments

    # Iterate over individuals in the given random order
    for (tt,ii) in economy.ordering:
        individual = economy.individuals[tt][ii]
        loc = economy.locs[tt] #locs[tt] is ordered by preference 
        for (jj,ll) in enumerate(loc):
            if ll not in full_locations: # -> the individual is offered here
                offers[tt] = add1_at(offers[tt], jj)
                if jj < individual.nlocs_considered: # -> the individual is vaccinated here
                    assignments[tt] = add1_at(assignments[tt], jj)
                    economy.occupancies[ll] += 1
                    if economy.occupancies[ll] == capacity:
                        full_locations.add(ll)
                break

    print("Finished assigning individuals in:", round(time.time() - time1, 2), "seconds")
    return



def assignment_stats(economy: Economy):
    total_pop = np.sum([len(economy.individuals[tt]) for tt in range(economy.n_geogs)])
    # offers
    offers = economy.offers
    print("Offers:")
    for ii in range(5):
        offers_ii = [offers[tt][ii] for tt in range(economy.n_geogs) if ii < len(offers[tt])]
        print(f"% Rank {ii} offers: {np.sum(offers_ii)/total_pop*100}")

    frac_offered_any = np.sum([np.sum(offers[tt]) for tt in range(economy.n_geogs)]) / total_pop
    print(f"% Offered: {frac_offered_any * 100}")
    max_rank_offered = np.max([len(offers[tt]) for tt in range(economy.n_geogs)])
    print(f"Max rank offered: {max_rank_offered}")


    # assignments
    assignments = economy.assignments
    print("Assignments:")
    for ii in range(5):
        assignments_ii = [assignments[tt][ii] for tt in range(economy.n_geogs) if ii < len(assignments[tt])]
        print(f"% Rank {ii} assignments: {np.sum(assignments_ii)/total_pop*100}")
    print(f"% Assigned: {np.sum([np.sum(assignments[tt]) for tt in range(economy.n_geogs)]) / total_pop * 100}")

    return frac_offered_any




def assignment_shares(
    df:pd.DataFrame,
    assignments:List[List[int]],
    cw_pop:pd.DataFrame, #must be sorted by blkid
    clip:Tuple[float, float] = (0.05, 0.95)
    ) -> pd.DataFrame:
    """
    Compute shares of each ZIP code from the block-level assignment
    Input: 
        assignments: returned by assignment (random_fcfs)
        df: DataFrame at the market level, need market_ids
        cw_pop: DataFrame at the geog level with columns market_ids, blkid, population
    Output: DataFrame at the market-level with 'shares' column replaced
    """
    cw_pop['assigned_pop'] = np.array([np.sum(ww) for ww in assignments])
    df_g = cw_pop.groupby('market_ids').agg({'population': 'sum', 'assigned_pop': 'sum'}).reset_index()
    df_g['shares'] = df_g['assigned_pop'] / df_g['population']
    df_g['shares'] = df_g['shares'].clip(clip[0], clip[1])
    print("Winsorized shares to ({}, {})".format(clip[0], clip[1]))
    df.drop(columns=['shares'], inplace=True)
    return df.merge(df_g[['market_ids', 'shares']], on='market_ids', how='left')



def subset_locs(
        offers:List[np.array], 
        agent_data:pd.DataFrame
        ) -> pd.DataFrame:
    """
    Subset the geog-level data to the locations that were actually offered
    """
    nlocs = [len(tt) for tt in offers] # number of locations that were assigned in each geog
    nlocs_df = pd.DataFrame({'blkid': np.unique(agent_data.blkid), 'nlocs': nlocs}) #relies on np.unique(agent_data.blkid) being in the same order as all the lists in economy
    agent_data_tosub = agent_data.merge(nlocs_df, on='blkid', how='left')
    agent_data_subset = agent_data_tosub.groupby('blkid').apply(lambda x: x.head(x['nlocs'].iloc[0])).reset_index(drop=True)
    agent_data_subset = agent_data_subset.drop(columns=['nlocs'])
    agent_data_subset = agent_data_subset.assign(
        weights = np.concatenate(offers)
    )
    # drop zero weights
    agent_data_subset = agent_data_subset[agent_data_subset.weights != 0]
    # normalize weights
    agent_data_subset = agent_data_subset.assign(
        weights = agent_data_subset['weights'] / agent_data_subset.groupby('market_ids')['weights'].transform('sum')
    )

    return agent_data_subset
