from vax_entities import Individual, Geog, Location

import numpy as np
from scipy.stats import gumbel_r
from typing import List, Optional
import random

def initialize(distmatrix: np.ndarray, #n_geogs x n_locations
               locmatrix: np.ndarray, #n_geogs x n_locations
               distcoef: np.ndarray, #length = number of geogs
               abd: np.ndarray, #length = number of geogs
               hpi: List[int], #length = number of geogs
               capacity: int = 10000, 
               M: int = 10, 
               n_individuals: Optional[List[int]] = None, # number of individuals in each geog
               seed: int = 1234):


    n_geogs = len(abd)

    np.random.seed(seed)

    # precompute ab_epsilon
    ab_epsilon = abd[:, np.newaxis] + distcoef.reshape(-1, 1) * distmatrix

    # Create Geog objects
    geogs = [
        Geog(
            tt, 
            hpi[tt], 
            distmatrix[tt, :],
            distcoef[tt], 
            abd[tt], 
            ab_epsilon[tt, :],
            [Individual(geog_id=tt, 
                        epsilon_ij=gumbel_r.rvs(size=M),
                        u_ij=np.zeros(M)
                        ) for ii in range(n_individuals[tt])],
            locmatrix[tt, :]) 
        for tt in range(n_geogs)]

    print(f"Created {n_geogs} geogs.")

    # Compute utility and ranking for each individual
    compute_ranking(geogs)
    print("Done computing ranking.")

    # Create Location objects
    n_locations = distmatrix.shape[0]
    locations = [Location(ll, capacity) for ll in range(n_locations)]

    return geogs, locations


def compute_ranking(geogs: List[Geog]):
    M = len(geogs[0].location_ids)
    for tt in geogs:
        for ii in tt.individuals:
            ii.u_ij = tt.ab_epsilon + ii.epsilon_ij
            ii.location_ranking = np.argsort(ii.u_ij)[::-1]  # reversed for descending order
            ii.locations_ranked = [tt.location_ids[index] for index in ii.location_ranking]


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


    # Iterate over individuals in random order
    for tt,ii in ordering:
        for ll in ii.locations_ranked:
            if locations[ll].capacity > locations[ll].occupancy:
                locations[ll].occupancy += 1
                ii.location = ll
                break

    if report:
        assignment_stats(geogs, locations)

def sequential(geogs: List[Geog], locations: List[Location], seed=1234, report=True, reset=True):

    np.random.seed(seed)

    if reset:
        reset_assignments(geogs, locations)
    
    M = len(geogs[0].location_ids)
    for round in range(M):
        individuals_remaining = shuffle_list([individual for geog in geogs for individual in geog.individuals if individual.location == 0])
        if report:
            print("Round: ", round, ", Individuals remaining: ", len(individuals_remaining))
        for ii in individuals_remaining:
            ll = ii.locations_ranked[round]
            if locations[ll].capacity > locations[ll].occupancy:
                locations[ll].occupancy += 1
                ii.location = ll

    if report:
        assignment_stats(geogs, locations)

def reset_assignments(geogs: List[Geog], locations: List[Location]):
    for geog in geogs:
        for individual in geog.individuals:
            individual.location = 0
    for location in locations:
        location.occupancy = 0

def assignment_stats(geogs: List[Geog], locations: List[Location] = None):
    n_individuals = sum(len(geog.individuals) for geog in geogs)
    print("********\nReporting stats for {} individuals in {} geogs".format(n_individuals, len(geogs)))

    M = len(geogs[0].location_ids)
    for cc in range(M):
        n_assigned = sum(ii.locations_ranked[cc] == ii.location for geog in geogs for ii in geog.individuals)
        print("Fraction assigned to choice {}: {:.3f}".format(cc, n_assigned / n_individuals))

    n_unassigned = sum(ii.location == 0 for geog in geogs for ii in geog.individuals)
    print("Fraction unassigned: {:.3f}".format(n_unassigned / n_individuals))

    if locations:
        print("********\nReporting stats for {} locations with capacity {}".format(len(locations), locations[0].capacity))
        occupancies = [loc.occupancy for loc in locations]
        print("\nMean occupancy: {:.3f}".format(np.mean(occupancies)))
        print("\nFraction of locations with occupancy > 0: {:.3f}".format(np.mean([occ > 0 for occ in occupancies])))
        occ_quantiles = np.quantile(occupancies, np.linspace(0, 1, 11))
        print("\nOccupancy quantiles: ")
        for q, occ in zip(np.linspace(0, 1, 11), occ_quantiles):
            print("{:.1f}: {:.2f}".format(q, occ))

