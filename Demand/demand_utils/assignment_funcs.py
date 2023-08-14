# assignment_funcs.py
# functions to assign individuals to locations

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
    assert len(distcoefs) == economy.n_geogs == len(abd)
    # compute all-but-epsilon for each location for each geography
    economy.abepsilon = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)] 
    time2 = time.time()
    print("Computed abepsilon in:", round(time2- time1, 2), "seconds.\nAssigning individuals...")

    # reset occupancies 
    economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
    full_locations = set()
    # reset offers and assignments
    economy.offers = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]
    economy.assignments = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]

    # Iterate over individuals in the shuffled ordering
    for (tt,ii) in economy.ordering:
        for (jj,ll) in enumerate(economy.locs[tt]): #locs[tt] is ordered by distance from geography tt, in ascending order
            if ll not in full_locations or jj==len(economy.locs[tt])-1:
                # -> the individual is offered here
                economy.offers[tt][jj] += 1
                if economy.abepsilon[tt][jj] > economy.epsilon_diff[tt][ii]: # -> the individual is vaccinated here
                    economy.assignments[tt][jj] += 1
                    economy.occupancies[ll] += 1
                    if economy.occupancies[ll] == capacity:
                        full_locations.add(ll)
                break

    print("Assigned individuals in:", round(time.time() - time2, 2), "seconds")
    return



def seq_lotteries(economy: Economy,
                  distcoefs: np.ndarray,
                  abd: np.ndarray,
                  capacity: int
                  ): #TODO: test this
    """
    In each round rr, assign individuals to their ranked rr-th location, breaking ties randomly if capacity is reached.
    """
    time1 = time.time()
    # compute all-but-epsilon for each location for each geography
    economy.abepsilon = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)]
    time2 = time.time()
    print("Computed abepsilon in:", round(time2- time1, 2), "seconds.\nAssigning individuals...")

    # reset occupancies
    economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
    full_locations = set()

    indivs_rem = economy.ordering.copy()
    rank = 0
    while indivs_rem:
        np.random.shuffle(indivs_rem)
        for (tt,ii) in indivs_rem:
            ll = economy.locs[tt][rank]
            if ll not in full_locations:  # -> the individual is offered here
                economy.offers[tt][rank] += 1
                indivs_rem.remove((tt,ii))
                if economy.abepsilon[tt][rank] > economy.epsilon_diff[tt][ii]: # -> the individual is vaccinated here
                    economy.assignments[tt][rank] += 1
                    economy.occupancies[ll] += 1
                    if economy.occupancies[ll] == capacity:
                        full_locations.add(ll)
        rank += 1

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


def assignment_stats(economy: Economy, max_rank: int = 10):
    total_pop = np.sum([len(economy.epsilon_diff[tt]) for tt in range(economy.n_geogs)])
    
    # offers
    offers = economy.offers
    print("Offers:")
    for ii in range(max_rank):
        offers_ii = [offers[tt][ii] for tt in range(economy.n_geogs) if ii < len(offers[tt])]
        sum_offers_ii = np.sum(offers_ii)
        print(f"% Rank {ii} offers: {sum_offers_ii/total_pop*100}")

    frac_offered_any = np.sum([np.sum(offers[tt]) for tt in range(economy.n_geogs)]) / total_pop
    print(f"% Offered: {frac_offered_any * 100}")
    if frac_offered_any < 1:
        print("*********\nWarning: not all individuals are offered") #shouldn't happen since we offer the last location
    max_rank_offered = np.max([np.max(np.flatnonzero(offers[tt])) for tt in range(economy.n_geogs)])
    print(f"Max rank offered: {max_rank_offered}")
    # number of individuals offered max_rank_offered
    print(f"Number of individuals offered max_rank_offered: {np.sum([offers[tt][max_rank_offered] for tt in range(economy.n_geogs) if max_rank_offered < len(offers[tt])])}")


    # assignments
    assignments = economy.assignments
    print("Assignments:")
    for ii in range(max_rank):
        assignments_ii = [assignments[tt][ii] for tt in range(economy.n_geogs) if ii < len(assignments[tt])]
        print(f"% Rank {ii} assignments: {np.sum(assignments_ii)/total_pop*100}")
    print(f"% Assigned: {np.sum([np.sum(assignments[tt]) for tt in range(economy.n_geogs)]) / total_pop * 100}")

    return frac_offered_any

