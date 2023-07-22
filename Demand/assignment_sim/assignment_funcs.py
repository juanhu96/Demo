from vax_entities import Individual, Geog
# , Location

import numpy as np
from scipy.stats import gumbel_r
from typing import List, Optional
import random
import time
from multiprocessing import Pool
import os
from concurrent.futures import ProcessPoolExecutor




def initialize_geogs(distmatrix: np.ndarray, #n_geogs x M
               locmatrix: np.ndarray, #n_geogs x M
               distcoef: np.ndarray, #length = number of geogs
               abd: np.ndarray, #length = number of geogs
               M: int = 10, 
               n_individuals: Optional[List[int]] = None, # number of individuals in each geog
               seed: int = 1234):


    n_geogs = distmatrix.shape[0]

    np.random.seed(seed)

    # precompute ab_epsilon
    ab_epsilon = abd[:, np.newaxis] + distcoef.reshape(-1, 1) * distmatrix
    print(f"Computed ab_epsilon.")

    # Generate random epsilon_ij (first one is outside option)
    epsilon_ij = [gumbel_r.rvs(size=(n_individuals[tt], M+1)) for tt in range(n_geogs)]
    print(f"Generated epsilon_ij.")

    # Create Geog objects
    geogs = [
        Geog(
            tt, 
            distmatrix[tt, :],
            distcoef[tt],
            ab_epsilon[tt, :],
            [Individual(geog_id=tt, 
                        epsilon_ij=epsilon_ij[tt][ii, :],
                        u_ij=np.zeros(M)
                        ) for ii in range(n_individuals[tt])],
            locmatrix[tt, :]) 
        for tt in range(n_geogs)]

    print(f"Created geogs.")

    # Compute utility and ranking for each individual
    compute_ranking(geogs)
    print("Done computing ranking.")

    # # Create Location objects
    # n_locations = len(np.unique(locmatrix))
    # locations = [Location(ll, capacity) for ll in range(n_locations)]

    return geogs


def shuffle_individuals(geogs: List[Geog]):
    indiv_ordering = [(tt,ii) for tt in range(len(geogs)) for ii in range(len(geogs[tt].individuals))]
    np.random.shuffle(indiv_ordering)
    return indiv_ordering


# rank locations in descending order, subset to u_ij greater than outside option (epsilon_ij[0])
def compute_geog_ranking(tt: Geog):
    for ii in tt.individuals:
        ii.u_ij = tt.ab_epsilon + ii.epsilon_ij[1:]
        inside_inds = np.argwhere(ii.u_ij > ii.epsilon_ij[0]).flatten()
        ii.location_ranking = inside_inds[np.argsort(ii.u_ij[inside_inds])][::-1]

def compute_ranking(geogs: List[Geog], n_processes: int = 32):
    with ProcessPoolExecutor(max_workers=min(n_processes, os.cpu_count())) as executor:
        list(executor.map(compute_geog_ranking, geogs))


def shuffle_list(original_list):
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list


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
        for (jj,ll) in enumerate(geogs[tt].individuals[ii].location_ranking):
            locid = geogs[tt].location_ids[ll]
            if occupancies[locid].occupancy < capacity:
                occupancies[locid].occupancy += 1
                geogs[tt].individuals[ii].location_assigned = locid
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

