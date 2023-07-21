
class Individual:
    def __init__(self, geog_id, epsilon_ij, u_ij):
        self.geog_id = geog_id
        self.epsilon_ij = epsilon_ij  # epsilon_ij[ll] is the random component of the utility of location ll
        self.u_ij = u_ij
        self.location_ranking = []
        self.location_assigned = -1  # location id of the location the individual is assigned to
        self.rank_assigned = -1  # rank of the location the individual is assigned to

class Geog:
    def __init__(self, id, dist, distcoef, ab_epsilon, individuals, location_ids):
        self.id = id
        self.dist = dist  # distance to each location
        self.distcoef = distcoef  # coefficient on distance term
        self.ab_epsilon = ab_epsilon  # ab_epsilon[ll] = abd + distcoef * dist[ll]
        self.individuals = individuals
        self.location_ids = location_ids

class Location:
    def __init__(self, id, capacity, occupancy=0, filled=False):
        self.id = id
        self.capacity = capacity
        self.occupancy = occupancy
        self.filled = filled

