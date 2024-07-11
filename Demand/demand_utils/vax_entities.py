import numpy as np
from scipy.stats import gumbel_r, logistic


class Economy:
    def __init__(
            self, 
            locs, # list of lists of location IDs, corresponding to dists
            dists, # list of lists of distances, sorted ascending
            geog_pops, # list of lengths of individuals in each geography
            max_rank = 300, # max rank to offer
            shuffle=True, # whether to shuffle the ordering of individuals
            epsilon_opt = "logistic", # "logistic", "gumbel", "zero"
            epsilon_scale = 1, # scale factor for epsilon
            mnl = False, # whether to use MNL
            seed = 1234 # random seed
            ):

        np.random.seed(seed)

        self.locs = locs # list of location IDs, sorted by distance within each geography
        self.dists = dists # list of distances, sorted by distance within each geography
        n_geogs = len(geog_pops)
        self.n_geogs = n_geogs
        self.total_pop = np.sum(geog_pops)

        self.ordering = [(tt,ii) for tt in range(n_geogs) for ii in range(geog_pops[tt])] # list of tuples of (geog, individual) indices
        if shuffle:
            np.random.shuffle(self.ordering)

        self.abepsilon = [np.zeros(len(locs[tt])) for tt in range(n_geogs)]
        # all-but-epsilon,  n_geogs x n_locs. abepsilon[tt][ll] = abd[tt] + distcoef * dists[tt][ll]. 

        self.offers = [np.concatenate([[geog_pops[tt]], np.zeros(len(locs[tt])-1, dtype=int)]) for tt in range(n_geogs)]
        # list of lists of location rankings offered, length n_geogs. offers[tt][ll] = number of individuals in tt offered location ranked ll
        # initialize with all individuals offered their nearest location for the first demand estimation

        self.assignments = [np.zeros(len(locs[tt]), dtype=int) for tt in range(n_geogs)]
        # list of lists of locations assigned, length n_geogs. assignments[tt][ll] = number of individuals in tt assigned the location ranked ll
        
        locids = np.unique(np.concatenate(locs))

        self.occupancies = dict.fromkeys(np.unique(locids), 0) # number of individuals assigned to each location

        self.maxrank_geogs = [len(ll)-1 for ll in locs] # list of max rank that can be offered in each geography (mostly max_rank-1, but could be less if there are fewer than max_rank locations in a geography)
        self.fallback_loc = [ll[-1] for ll in locs] # list of fallback locations (last location in each geography)
        # TODO: only need one of the 2 above

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
        
        # For MNL:
        if mnl:
            nlocs = [len(locs[tt]) for tt in range(n_geogs)]
            self.gumbel_draws = [[gumbel_r.rvs(size=nlocs[tt]) for ii in range(geog_pops[tt])] for tt in range(n_geogs)]
            self.utils = [[np.zeros(nlocs[tt]) for ii in range(geog_pops[tt])] for tt in range(n_geogs)]
        else:
            self.gumbel_draws = None
            self.utils = None

        self.violation_count = [0 for ll in locids] # number of times each location has been overassigned
        self.agent_violations = np.zeros(n_geogs, dtype=int) # number of individuals who have been overassigned