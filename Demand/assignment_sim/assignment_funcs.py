from vax_entities import Individual, Geog
from vax_entities import Economy
# , Location

import numpy as np
from scipy.stats import gumbel_r, logistic
from typing import List, Optional, Tuple
import random
import time
from multiprocessing import Pool
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm




def initialize_economy(
        locs: List[np.ndarray], #pharmacy IDs in each geog (ordered)
        dists: List[np.ndarray], #sorted within each geog
        abd: np.ndarray, #length = number of geogs
        n_individuals, # number of individuals in each geog
        seed: int = 1234):

    time1 = time.time()
    print("Initializing economy...")
    np.random.seed(seed)
    n_geogs = len(locs)

    # Generate random epsilon_i
    epsilon_diff = [logistic.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
    # verified that it's equivalent to:
    # epsilon1 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
    # epsilon0 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)] # outside option
    # epsilon_diff = [epsilon0[tt] - epsilon1[tt] for tt in range(n_geogs)]

    # sort epsilon_diff (ascending) within each geog, for computing rank.
    # OK since we'll do assignments shuffled.
    epsilon_diff = [np.sort(epsilon_diff[tt]) for tt in range(n_geogs)]
    print("Finished generating epsilons:", time.time() - time1)


    economy = Economy(
        locs=locs,
        dists=dists,
        abd=abd,
        abe=np.zeros(shape=abd.shape),
        individuals = [
            [Individual(epsilon_diff[tt][ii]) for ii in range(n_individuals[tt])]
            for tt in range(n_geogs)
        ]
    )

    print("Finished initializing economy:", time.time() - time1)
    
    return economy



# condition : ii.u_ij > ii.epsilon0
# condition : ii.epsilon1 + ab_epsilon[ll] > ii.epsilon0
# condition : ii.epsilon0 - ii.epsilon1 < ab_epsilon[ll]
# condition : ii.epsilon_diff < ab_epsilon[ll]


# ab_epsilon is sorted descending within each geog since distances are sorted ascending
# individuals in geog tt are sorted by epsilon_diff (ascending)

def compute_geog_ranking(args: Tuple[List[Individual], np.ndarray]):
    individuals, ab_epsilon = args
    start_index = 0
    for ii in individuals:
        start_index = np.searchsorted(ab_epsilon[start_index:], ii.epsilon_diff) + start_index
        ii.nlocs_considered = start_index
    return individuals


def compute_economy_ranking(economy: Economy, distcoefs, poolnum: int = 1):
    time1 = time.time()
    economy.abe = [economy.abd[tt] + distcoefs[tt] * economy.dists[tt] for tt in range(economy.n_geogs)]
    print("Computing rankings using {} processes...".format(poolnum))
    with Pool(poolnum) as p:
        economy.individuals = p.map(compute_geog_ranking, [(economy.individuals[tt], economy.abe[tt]) for tt in range(economy.n_geogs)])

    print("Finished computing rankings in:", time.time() - time1, "seconds")

        







def shuffle_individuals(individuals: List[Individual]):
    indiv_ordering = [(tt,ii) for tt in range(len(individuals)) for ii in range(len(individuals[tt]))]
    np.random.shuffle(indiv_ordering)
    return indiv_ordering


def random_fcfs(economy: Economy, 
                n_locations: int,
                capacity: int,
                ordering: List[int]):
    
    time1 = time.time()
    print("Assigning individuals...")

    occupancies = np.zeros(n_locations)

    # Iterate over individuals in the given random order
    for (it, (tt,ii)) in enumerate(ordering):
        for (jj,ll) in enumerate(economy.locs[tt][:economy.individuals[tt][ii].nlocs_considered]):
            if occupancies[ll] < capacity:
                occupancies[ll] += 1
                economy.individuals[tt][ii].location_assigned = ll
                economy.individuals[tt][ii].rank_assigned = jj #not needed if not reporting (?)
                break
        if it % 1000000 == 0:
            print(f"Assigned {it/1000000} million individuals out of {len(ordering)} in {round((time.time() - time1), 2)} seconds")
    print("Finished assigning individuals:", time.time() - time1)




def reset_assignments(individuals: List[Individual]):
    for tt in individuals:
        for ii in tt:
            ii.location_assigned = -1
            ii.rank_assigned = -1


def assignment_stats(individuals: List[Individual]):
    n_individuals = sum(len(ii) for ii in individuals)
    print("********\nReporting stats for {} individuals in {} geogs".format(n_individuals, len(individuals)))

    assigned_ranks = np.array([ii.rank_assigned for tt in individuals for ii in tt if ii.location_assigned != -1])
    print("\nAssigned ranks : ")
    rank_counts = np.bincount(assigned_ranks)
    for rr in range(20):
        print(f"Fraction assigned rank {rr}: {rank_counts[rr]/n_individuals}")
    print("Fraction unassigned: {:.3f}".format(np.mean(assigned_ranks == -1)))

