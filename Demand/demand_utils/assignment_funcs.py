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
                capacity: int,
                mnl: bool = False,
                evaluation: bool = False
                ):
    """
    Assign individuals to locations in random order, first-come-first-serve.
    """
    
    time1 = time.time()
    assert len(distcoefs) == economy.n_geogs == len(abd)
    # compute all-but-epsilon for each location for each geography
    economy.abepsilon = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)] 
    time2 = time.time()
    if mnl:
        for tt in range(economy.n_geogs):
            for ii in range(len(economy.utils[tt])):
                economy.utils[tt][ii] = economy.gumbel_draws[tt][ii] + economy.abepsilon[tt]

    print("Computed abepsilon in:", round(time2- time1, 3), "seconds.\nAssigning individuals...")

    # reset occupancies 
    economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
    full_locations = set()
    # reset offers and assignments
    economy.offers = [np.zeros(len(economy.locs[tt]), dtype=int) for tt in range(economy.n_geogs)]
    economy.assignments = [np.zeros(len(economy.locs[tt]), dtype=int) for tt in range(economy.n_geogs)]
    time3 = time.time()
    print("time3 - time2:", round(time3-time2, 3))
    # Iterate over individuals in the shuffled ordering
    for (tt,ii) in economy.ordering:
        if mnl: # locations in preference order (sorted by utils descending)
            preforder = np.argsort(economy.utils[tt][ii])[::-1]
        else: # locations in existing order (sorted by distance)
            preforder = np.arange(len(economy.locs[tt]))

        for (jj,ll_ind) in enumerate(preforder):
            ll = economy.locs[tt][ll_ind]
            if evaluation: 
                offer_condition = ll not in full_locations
            else:
                offer_condition = ll not in full_locations or jj==len(preforder)-1
            if offer_condition:
                if ll in full_locations and jj==len(preforder)-1:
                    economy.violation_count[ll] += 1
                economy.offers[tt][jj] += 1
                if economy.abepsilon[tt][jj] > economy.epsilon_diff[tt][ii]: # -> the individual is vaccinated here
                    economy.assignments[tt][jj] += 1
                    economy.occupancies[ll] += 1
                    if economy.occupancies[ll] == capacity:
                        full_locations.add(ll)
                break
    time4 = time.time()
    print("time4 - time3:", round(time4-time3, 3))
    print("Number of full locations:", len(full_locations), " out of ", len(economy.occupancies))
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
    frac_offered_any = 0
    print("Offers:")
    for ii in range(max_rank):
        offers_ii = [offers[tt][ii] for tt in range(economy.n_geogs) if ii < len(offers[tt])]
        sum_offers_ii = np.sum(offers_ii)
        frac_offered_ii = sum_offers_ii/total_pop
        print(f"% Rank {ii} offers: {frac_offered_ii * 100:.5f}")
        frac_offered_any += frac_offered_ii

    print(f"% Offered any: {frac_offered_any * 100}")
    max_rank_offered = np.max([np.max(np.flatnonzero(offers[tt])) for tt in range(economy.n_geogs)])
    print(f"Max rank offered: {max_rank_offered}")
    print(f"Number of individuals offered max_rank_offered: {np.sum([offers[tt][max_rank_offered] for tt in range(economy.n_geogs) if max_rank_offered < len(offers[tt])])}")

    # offer distances
    offer_dists = np.concatenate([np.repeat(economy.dists[tt], offers[tt]) for tt in range(economy.n_geogs)])
    print(f"Mean offer distance: {np.mean(offer_dists):.5f}")
    # quantiles
    print("Quantiles of offered distances:")
    for qq in np.arange(0,1.1,0.1):
        print(f"{qq:.1f} quantile: {np.quantile(offer_dists, qq):.5f}")

    # assignments
    assignments = economy.assignments
    print("Assignments:")
    for ii in range(max_rank):
        assignments_ii = [assignments[tt][ii] for tt in range(economy.n_geogs) if ii < len(assignments[tt])]
        print(f"% Rank {ii} assignments: {np.sum(assignments_ii)/total_pop*100:.5f}")
    print(f"% Assigned: {np.sum([np.sum(assignments[tt]) for tt in range(economy.n_geogs)]) / total_pop * 100:.5f}")

    # assignment distances
    assignment_dists = np.concatenate([np.repeat(economy.dists[tt], assignments[tt]) for tt in range(economy.n_geogs)])
    print(f"Mean assignment distance: {np.mean(assignment_dists):.5f}")
    # quantiles
    print("Quantiles of assignment distances:")
    for qq in np.arange(0,1.1,0.1):
        print(f"{qq:.1f} quantile: {np.quantile(assignment_dists, qq):.5f}")

    print("Violations:")
    print("Max violation:", max(economy.violation_count))
    print("Number of locations with violations:", np.sum([1 for vv in economy.violation_count if vv > 0]))
    print("Number of individuals with violations:", np.sum(economy.violation_count))
    print("Mean violation (conditional on violation):", np.mean([vv for vv in economy.violation_count if vv > 0]))
    print("Median violation (conditional on violation):", np.median([vv for vv in economy.violation_count if vv > 0]))
        
    return frac_offered_any


# ===============================================================================
# ======================= Jingyuan: for evaluation only =========================
# ===============================================================================

def random_fcfs_eval(economy: Economy,
                    distcoefs: np.ndarray,
                    abd: np.ndarray,
                    capacity: int
                    ):
    """
    A modified version where the not all inidividuals are offered (POLICY EVALUATION ONLY)
    """
    
    time1 = time.time()
    assert len(distcoefs) == economy.n_geogs == len(abd)
    economy.abepsilon = [abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)] 
    time2 = time.time()
    print("Computed abepsilon in:", round(time2- time1, 2), "seconds.\nAssigning individuals...")

    economy.occupancies = dict.fromkeys(economy.occupancies.keys(), 0)
    full_locations = set()
    economy.offers = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]
    economy.assignments = [np.zeros(len(economy.locs[tt])) for tt in range(economy.n_geogs)]

    time3 = time.time()
    for (tt,ii) in economy.ordering:
        for (jj,ll) in enumerate(economy.locs[tt]):
            if ll not in full_locations: # NOTE: THE ONLY DIFF
                economy.offers[tt][jj] += 1
                if economy.abepsilon[tt][jj] > economy.epsilon_diff[tt][ii]: # -> the individual is vaccinated here
                    economy.assignments[tt][jj] += 1
                    economy.occupancies[ll] += 1
                    if economy.occupancies[ll] == capacity:
                        full_locations.add(ll)
                break
    
    time4 = time.time()
    print("Assigned individuals in:", round(time4 - time3, 2), "seconds")

    return
