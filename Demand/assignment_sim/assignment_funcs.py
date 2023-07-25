from vax_entities import Economy, Individual

import numpy as np
from scipy.stats import gumbel_r, logistic
from typing import List, Optional, Tuple
import random
import time
from multiprocessing import Pool



def initialize_economy(
        locs: List[np.ndarray], #pharmacy IDs in each geog (ordered)
        dists: List[np.ndarray], #sorted within each geog
        abd: np.ndarray, #length = number of geogs
        n_individuals, # number of individuals in each geog
        seed: int = 1234,
        epsilon_opt = "gumbel" # "logistic" or "gumbel" or "zero"
        ):

    time1 = time.time()
    print("Initializing economy...")
    np.random.seed(seed)
    n_geogs = len(locs)

    # Generate random epsilon_diff = epsilon0 - epsilon1
    if epsilon_opt == "gumbel":
        epsilon_0 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
        epsilon_1 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
        epsilon_diff = [epsilon_0[tt] - epsilon_1[tt] for tt in range(n_geogs)]
    elif epsilon_opt == "logistic":
        epsilon_diff = [logistic.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
    elif epsilon_opt == "zero":
        epsilon_diff = [-gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
    
    # logistic.rvs should be equivalent to gumbel_r.rvs - gumbel_r.rvs

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

    for ii in individuals:
        ii.nlocs_considered = np.searchsorted(-ab_epsilon, -ii.epsilon_diff) 
    return individuals


def compute_economy_ranking(economy: Economy, distcoefs, poolnum: int = 1, scale: float = 1.0):
    time1 = time.time()
    economy.abe = [economy.abd[tt] + (distcoefs[tt] * economy.dists[tt]) for tt in range(economy.n_geogs)]
    economy.abe = [scale * economy.abe[tt] for tt in range(economy.n_geogs)]

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
                ordering: List[Tuple[int, int]]):
    
    time1 = time.time()
    print("Assigning individuals...")

    # Initialize occupancies and full locations
    occupancies = dict.fromkeys(range(n_locations), 0)
    full_locations = set()



    individuals = economy.individuals
    locs = economy.locs
    weights = [np.zeros(shape=(len(locs[tt]))) for tt in range(len(locs))] #frequency weights for agent_data

    # Iterate over individuals in the given random order
    for (it, (tt,ii)) in enumerate(ordering):
        if individuals[tt][ii].nlocs_considered == 0:
            continue

        for (jj,ll) in enumerate(locs[tt][:individuals[tt][ii].nlocs_considered]):
            # Check if location is not full and assign the individual to this location
            if ll not in full_locations:
                occupancies[ll] += 1
                individuals[tt][ii].location_assigned = ll
                individuals[tt][ii].rank_assigned = jj
                weights[tt][jj] += 1
                
                
                # If the location has reached capacity, add it to the set of full locations
                if occupancies[ll] == capacity:
                    full_locations.add(ll)
                break
        if it % 1000000 == 0:
            print(f"Assigned {it/1000000} million individuals out of {len(ordering)} in {round((time.time() - time1), 2)} seconds")
        print(f"Assigned {it} individuals out of {len(ordering)} in {round((time.time() - time1), 2)} seconds")
        return weights


def assignment_stats(individuals: List[Individual]):
    n_individuals = sum(len(ii) for ii in individuals)
    print("********\nReporting stats for {} individuals in {} geogs".format(n_individuals, len(individuals)))

    assigned_ranks = np.array([ii.rank_assigned for tt in individuals for ii in tt])
    print("\nAssigned ranks : ")
    assigned_ind = assigned_ranks >= 0
    rank_counts = np.bincount(assigned_ranks[assigned_ind])
    max_rank = np.max(assigned_ranks[assigned_ind])
    print("Max rank assigned: ", max_rank)
    for rr in range(10):
        print(f"% assigned rank {rr}: {round(rank_counts[rr]/n_individuals*100, 2)}")
    print(f"% unassigned: {round(np.sum(~assigned_ind)/n_individuals*100, 2)}")

