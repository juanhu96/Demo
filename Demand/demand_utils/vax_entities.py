import numpy as np

class Individual:
    def __init__(self, epsilon_diff):
        self.epsilon_diff = epsilon_diff # epsilon0 - epsilon1
        self.nlocs_considered = -1  # number of locations in consideration set
        # self.location_assigned = -1  # location id of the location the individual is assigned to
        self.assigned = False
        self.loc_offered = -1  # rank of the location the individual is offered

class Economy:
    def __init__(self, locs, dists, individuals):
        self.locs = locs # list of location IDs sorted by distance
        self.dists = dists # list of distances sorted by distance
        self.individuals = individuals # list of lists of individuals, length n_geogs
        self.abd = None # all-but-distance (and epsilon), length n_geogs
        self.abe = None # all-but-epsilon,  n_geogs x n_locs. abe[tt][ll] = abd[tt] + distcoef * dists[tt][ll]
        self.offers = None # list of lists of location rankings offered, length n_geogs. offers[tt][ll] = number of individuals in tt offered location ranked ll
        self.assignments = None # list of lists of locations assigned, length n_geogs. assignments[tt][ll] = number of individuals in tt assigned the location ranked ll
        self.n_geogs = len(locs)

