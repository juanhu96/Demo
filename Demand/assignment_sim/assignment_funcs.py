from vax_entities import Individual, Geog, Location

import numpy as np
from scipy.stats import gumbel_r
from typing import List, Optional
import random

def initialize(distmatrix: np.ndarray, #n_geogs x M
               locmatrix: np.ndarray, #n_geogs x M
               distcoef: np.ndarray, #length = number of geogs
               abd: np.ndarray, #length = number of geogs
               capacity: int = 10000, 
               M: int = 10, 
               n_individuals: Optional[List[int]] = None, # number of individuals in each geog
               seed: int = 1234):


    n_geogs = distmatrix.shape[0]

    np.random.seed(seed)

    # precompute ab_epsilon
    ab_epsilon = abd[:, np.newaxis] + distcoef.reshape(-1, 1) * distmatrix
    print(f"Computed ab_epsilon.")

    # Generate random epsilon_ij
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

    # Create Location objects
    n_locations = len(np.unique(locmatrix))
    locations = [Location(ll, capacity) for ll in range(n_locations)]

    return geogs, locations


def compute_ranking(geogs: List[Geog]):
    for tt in geogs:
        for ii in tt.individuals:
            ii.u_ij = tt.ab_epsilon + ii.epsilon_ij[1:]
            ii.location_ranking = np.argsort(ii.u_ij)[::-1][:sum(ii.u_ij > ii.epsilon_ij[0])]
              # descending order, subset to utilities greater than outside option (epsilon_ij[0])


def shuffle_list(original_list):
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list


def random_fcfs(geogs: List[Geog], 
                locations: List[Location], 
                ordering: List[int],
                report=True, 
                reset=True):

    if reset:
        reset_assignments(geogs, locations)
    # Iterate over individuals in the given random order
    for (tt,ii) in ordering:
        for (jj,ll) in enumerate(geogs[tt].individuals[ii].location_ranking):
            locid = geogs[tt].location_ids[ll]
            if locations[locid].capacity > locations[locid].occupancy:
                locations[locid].occupancy += 1
                geogs[tt].individuals[ii].location_assigned = locid
                geogs[tt].individuals[ii].rank_assigned = jj
                break

    if report:
        assignment_stats(geogs, locations)

def sequential(geogs: List[Geog], locations: List[Location], seed=1234, report=True, reset=True):

    np.random.seed(seed)

    if reset:
        reset_assignments(geogs, locations)
    
    M = len(geogs[0].location_ids)
    for round in range(M):
        individuals_remaining = shuffle_list([individual for geog in geogs for individual in geog.individuals if individual.location_assigned == 0])
        if report:
            print("Round: ", round, ", Individuals remaining: ", len(individuals_remaining))
        for ii in individuals_remaining:
            ll = ii.locations_ranked[round] #TODO:no more locations_ranked
            if locations[ll].capacity > locations[ll].occupancy:
                locations[ll].occupancy += 1
                ii.location_assigned = ll

    if report:
        assignment_stats(geogs, locations)


def reset_assignments(geogs: List[Geog], locations: List[Location]):
    for geog in geogs:
        for individual in geog.individuals:
            individual.location_assigned = -1
    for location in locations:
        location.occupancy = 0


def assignment_stats(geogs: List[Geog], locations: List[Location] = None):
    n_individuals = sum(len(geog.individuals) for geog in geogs)
    print("********\nReporting stats for {} individuals in {} geogs".format(n_individuals, len(geogs)))

    M = len(geogs[0].location_ids)
    assigned_ranks = [ii.rank_assigned for geog in geogs for ii in geog.individuals]
    print("\nAssigned ranks : ")
    for rank in range(M):
        n_assigned = sum(assigned_ranks == rank)
        print("{:.1f}: {:.3f}".format(rank, n_assigned / n_individuals))

    print("Fraction unassigned: {:.3f}".format(sum(assigned_ranks == -1) / n_individuals))

    if locations:
        print("********\nReporting stats for {} locations with capacity {}".format(len(locations), locations[0].capacity))
        occupancies = [loc.occupancy for loc in locations]
        print("\nMean occupancy: {:.3f}".format(np.mean(occupancies)))
        print("\nFraction of locations with occupancy > 0: {:.3f}".format(np.mean([occ > 0 for occ in occupancies])))
        occ_quantiles = np.quantile(occupancies, np.linspace(0, 1, 11))
        print("\nOccupancy quantiles: ")
        for q, occ in zip(np.linspace(0, 1, 11), occ_quantiles):
            print("{:.1f}: {:.0f}".format(q, occ))

