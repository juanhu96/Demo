import numpy as np
from scipy.stats import gumbel_r, logistic


class Economy:
    def __init__(
            self, 
            locs, # list of lists of location IDs, corresponding to dists
            dists, # list of lists of distances, sorted ascending
            geog_pops, # list of lengths of individuals in each geography
            max_rank = 100, # max rank to offer
            shuffle=True, # whether to shuffle the ordering of individuals
            epsilon_opt = "logistic", # "logistic", "gumbel", "zero"
            epsilon_scale = 1, # scale factor for epsilon
            seed = 1234 # random seed
            ):

        np.random.seed(seed)

        self.locs = [ll[:max_rank] for ll in locs] # list of location IDs, sorted by distance within each geography
        self.dists = [dd[:max_rank] for dd in dists] # list of distances, sorted by distance within each geography
        n_geogs = len(geog_pops)
        self.n_geogs = n_geogs
        self.total_pop = np.sum(geog_pops)

        self.ordering = [(tt,ii) for tt in range(n_geogs) for ii in range(geog_pops[tt])] # list of tuples of (geog, individual) indices
        if shuffle:
            np.random.shuffle(self.ordering)

        self.abepsilon = [np.zeros(max_rank) for tt in range(n_geogs)] 
        # all-but-epsilon,  n_geogs x n_locs. abepsilon[tt][ll] = abd[tt] + distcoef * dists[tt][ll]. 

        self.offers = [np.zeros(max_rank) for tt in range(n_geogs)]
        # list of lists of location rankings offered, length n_geogs. offers[tt][ll] = number of individuals in tt offered location ranked ll

        self.assignments = [np.zeros(max_rank) for tt in range(n_geogs)]
        # list of lists of locations assigned, length n_geogs. assignments[tt][ll] = number of individuals in tt assigned the location ranked ll
        
        locids = np.unique(np.concatenate(locs))

        self.occupancies = dict.fromkeys(np.unique(locids), 0) # number of individuals assigned to each location

        # generate epsilon_diff - based on the comparison to empirical shares with nearest distance, i'm pretty sure it should be epsilon_opt="logistic" and scale=1
        if epsilon_opt == "gumbel":
            epsilon_0 = [gumbel_r.rvs(size=(geog_pops[tt])) for tt in range(n_geogs)]
            epsilon_1 = [gumbel_r.rvs(size=(geog_pops[tt])) for tt in range(n_geogs)]
            epsilon_diff = [epsilon_0[tt] - epsilon_1[tt] for tt in range(n_geogs)]
        elif epsilon_opt == "logistic":
            epsilon_diff = [logistic.rvs(size=(geog_pops[tt])) for tt in range(n_geogs)]
        elif epsilon_opt == "zero":
            epsilon_diff = [-gumbel_r.rvs(size=(geog_pops[tt])) for tt in range(n_geogs)]
        else:
            raise ValueError("epsilon_opt must be 'gumbel', 'logistic', or 'zero'")

        if epsilon_scale != 1:
            epsilon_diff = [epsilon_scale * epsilon_diff[tt] for tt in range(n_geogs)]

        self.epsilon_diff = epsilon_diff 
