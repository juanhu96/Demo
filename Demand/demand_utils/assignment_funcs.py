try:
    from demand_utils.vax_entities import Economy
except:
    from Demand.demand_utils.vax_entities import Economy

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import time
# from multiprocessing import Pool


# condition : u_ij > epsilon0
# <=> epsilon1 + abepsilon[ll] > epsilon0
# <=> abepsilon[ll] > epsilon0 - epsilon1
# <=> abepsilon[ll] > epsilon_diff

# abepsilon is sorted descending within each geog since distances are sorted ascending

def random_fcfs(economy: Economy,
                distcoefs: np.ndarray,
                abd: np.ndarray,
                capacity: int
                ):
    """
    Assign individuals to locations in random order, first-come-first-serve.
    """
    
    time1 = time.time()

    # compute all-but-epsilon for each location for each geography
    economy.abepsilon = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)] 
    time2 = time.time()
    print("Computed abepsilon in:", round(time2- time1, 2), "seconds.\nAssigning individuals...")

    # reset occupancies 
    economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
    full_locations = set()

    # Iterate over individuals in the shuffled ordering
    for (tt,ii) in economy.ordering:
        for (jj,ll) in enumerate(economy.locs[tt]): #locs[tt] is ordered by distance from geography tt, in ascending order
            if ll not in full_locations: # -> the individual is offered here
                economy.offers[tt][jj] += 1
                if economy.abepsilon[tt][jj] > economy.epsilon_diff[tt][ii]: # -> the individual is vaccinated here
                    economy.assignments[tt][jj] += 1
                    economy.occupancies[ll] += 1
                    if economy.occupancies[ll] == capacity:
                        full_locations.add(ll)
                break

    print("Assigned individuals in:", round(time.time() - time2, 2), "seconds")
    return


def pref_stats(economy: Economy):
    """
    Compute statistics for the preferences.
    """
    total_pop = np.sum([len(economy.epsilon_diff[tt]) for tt in range(economy.n_geogs)])
    print(f"Total population: {total_pop}")

    # abepsilon of the nearest location for each geography
    abepsilon_nearest = [economy.abepsilon[tt][0] for tt in range(economy.n_geogs)]

    pop_willing = np.sum([np.sum(economy.epsilon_diff[tt] < abepsilon_nearest[tt]) for tt in range(economy.n_geogs)])

    print(f"Percentage of individuals willing to be vaccinated at the nearest location: {round(pop_willing /total_pop * 100, 3)}%")
    return


def assignment_stats(economy: Economy):
    total_pop = np.sum([len(economy.epsilon_diff[tt]) for tt in range(economy.n_geogs)])
    
    # offers
    offers = economy.offers
    print("Offers:")
    for ii in range(5):
        offers_ii = [offers[tt][ii] for tt in range(economy.n_geogs) if ii < len(offers[tt])]
        print(f"% Rank {ii} offers: {np.sum(offers_ii)/total_pop*100}")

    frac_offered_any = np.sum([np.sum(offers[tt]) for tt in range(economy.n_geogs)]) / total_pop
    print(f"% Offered: {frac_offered_any * 100}")
    if frac_offered_any < 1:
        print("*********\nWarning: not all individuals are offered")
    max_rank_offered = np.max([np.max(np.flatnonzero(offers[tt])) for tt in range(economy.n_geogs)])
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
    clip:Tuple[float, float] = (0.05, 0.95) #clip shares to be between 0.05 and 0.95
    ) -> pd.DataFrame:
    """
    Compute shares of each ZIP code from the block-level assignment
    Input: 
        assignments: economy.assignment after random_fcfs
        df: DataFrame at the market level, need market_ids
        cw_pop: DataFrame at the geog level with columns market_ids, blkid, population
    Output: DataFrame at the market-level with 'shares' column replaced
    """
    # check that cw_pop is sorted by blkid
    # assert cw_pop['blkid'].is_monotonic_increasing
    
    cw_pop['assigned_pop'] = np.array([np.sum(ww) for ww in assignments])
    df_g = cw_pop.groupby('market_ids').agg({'population': 'sum', 'assigned_pop': 'sum'}).reset_index()
    df_g['shares'] = df_g['assigned_pop'] / df_g['population']
    
    # winsorize shares for demand estimation
    df_g['shares'] = df_g['shares'].clip(clip[0], clip[1])

    # replace shares in df
    df.drop(columns=['shares'], inplace=True)
    return df.merge(df_g[['market_ids', 'shares']], on='market_ids', how='left')
