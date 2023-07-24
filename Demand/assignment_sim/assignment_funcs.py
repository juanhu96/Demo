from vax_entities import Individual, Geog
# , Location

import numpy as np
from scipy.stats import gumbel_r
from typing import List, Optional, Tuple
import random
import time
from multiprocessing import Pool
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm




def initialize_geogs(
        locs: List[np.ndarray], #pharmacies in each geog
        dists: List[np.ndarray], #sorted within each geog
        abd: np.ndarray, #length = number of geogs
        n_individuals, # number of individuals in each geog
        seed: int = 1234):


    n_geogs = len(locs)
    np.random.seed(seed)
    # Generate random epsilon_i
    epsilon1 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)]
    epsilon0 = [gumbel_r.rvs(size=(n_individuals[tt])) for tt in range(n_geogs)] # outside option
    epsilon_diff = [epsilon0[tt] - epsilon1[tt] for tt in range(n_geogs)]

    # sort epsilon_diff (ascending) within each geog, for computing rank.
    # OK since we'll do assignments shuffled.
    epsilon_diff = [np.sort(epsilon_diff[tt]) for tt in range(n_geogs)]
    print("Finished generating epsilons.")

    geogs = [
        Geog(
            location_ids=locs[tt],
            distances=dists[tt],
            abd=abd[tt],
            individuals=[
                Individual(epsilon_diff[tt][ii])
                for ii in range(n_individuals[tt])
            ]
        )
        for tt in range(n_geogs)
    ]
    return geogs



# condition : ii.u_ij > ii.epsilon0
# condition : ii.epsilon1 + ab_epsilon[ll] > ii.epsilon0
# condition : ii.epsilon0 - ii.epsilon1 < ab_epsilon[ll]
# condition : -(ii.epsilon_diff) < ab_epsilon[ll]


# ab_epsilon is sorted descending within each geog since distances are sorted ascending
#individuals in geog tt are sorted by epsilon_diff (ascending)
def compute_geog_ranking(args: Tuple[Geog, float]):
    tt, distcoef = args
    tt.ab_epsilon = tt.abd + (distcoef * tt.distances)

    start_index = 0
    for ii in tt.individuals:
        start_index = np.searchsorted(tt.ab_epsilon[start_index:], -(ii.epsilon_diff)) + start_index
        ii.nlocs_considered = start_index


def compute_ranking(geogs: List[Geog], distcoefs, poolnum: int = 1):
    if poolnum == 1:
        for tt in range(len(geogs)):
            compute_geog_ranking((geogs[tt], distcoefs[tt]))
            if tt % 50000 == 0:
                print(f"Finished computing rankings for {tt} geogs.")
    else:
        with ProcessPoolExecutor(max_workers=poolnum) as executor:
            list(executor.map(compute_geog_ranking, zip(geogs, distcoefs)))

    print(f"Finished computing rankings for {len(geogs)} geogs.")


def shuffle_individuals(geogs: List[Geog]):
    indiv_ordering = [(tt,ii) for tt in range(len(geogs)) for ii in range(len(geogs[tt].individuals))]
    np.random.shuffle(indiv_ordering)
    return indiv_ordering




def random_fcfs(geogs: List[Geog], 
                n_locations: int,
                capacity: int,
                ordering: List[int],
                report=True, 
                reset=True):
    time1 = time.time()
    if reset:
        reset_assignments(geogs)

    occupancies = np.zeros(n_locations)
    time2 = time.time()
    # Iterate over individuals in the given random order
    for (tt,ii) in ordering:
        for (jj,ll) in enumerate(geogs[tt].location_ids[:geogs[tt].individuals[ii].nlocs_considered]):
            if occupancies[ll].occupancy < capacity:
                occupancies[ll].occupancy += 1
                geogs[tt].individuals[ii].location_assigned = ll
                geogs[tt].individuals[ii].rank_assigned = jj #not needed if not reporting (?)
                break
    time3 = time.time()
    if report:
        assignment_stats(geogs)
        print(f"Time elapsed: {time3-time1}")
        if reset:
            print(f"Time elapsed (reset_assignments): {time2-time1}")
        time4 = time.time()
        print(f"Time elapsed (assignment_stats): {time4-time3}")



def reset_assignments(geogs: List[Geog]):
    for geog in geogs:
        for individual in geog.individuals:
            individual.location_assigned = -1
    # for location in locations:
    #     location.occupancy = 0


def assignment_stats(geogs: List[Geog]):
    n_individuals = sum(len(geog.individuals) for geog in geogs)
    print("********\nReporting stats for {} individuals in {} geogs".format(n_individuals, len(geogs)))

    M = len(geogs[0].location_ids)
    assigned_ranks = np.array([ii.rank_assigned for geog in geogs for ii in geog.individuals])
    print("\nAssigned ranks : ")
    rank_counts = np.bincount(assigned_ranks[assigned_ranks >= 0], minlength=M)
    for rank, count in enumerate(rank_counts):
        print("{:.1f}: {:.3f}".format(rank, count / n_individuals))

    print("Fraction unassigned: {:.3f}".format(np.mean(assigned_ranks == -1)))

