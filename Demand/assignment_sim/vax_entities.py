
class Individual:
    def __init__(self, geog_id, epsilon_ij, u_ij, location_ranking=None, locations_ranked=None, location=0):
        self.geog_id = geog_id
        self.epsilon_ij = epsilon_ij  # epsilon_ij[ll] is the random component of the utility of location ll
        self.u_ij = u_ij
        self.location_ranking = location_ranking if location_ranking else []
        self.locations_ranked = locations_ranked if locations_ranked else []  # locations_ranked[rr] is the location id of the rr-th choice
        self.location = location  # location id of the location the individual is assigned to

class Geog:
    def __init__(self, id, hpi, dist, distcoef, abd, ab_epsilon, individuals, location_ids):
        self.id = id
        self.hpi = hpi
        self.dist = dist  # distance to each location
        self.distcoef = distcoef  # coefficient on distance term
        self.abd = abd  # utility except distance term
        self.ab_epsilon = ab_epsilon  # ab_epsilon[ll] = abd + distcoef * dist[ll]
        self.individuals = individuals
        self.location_ids = location_ids

class Location:
    def __init__(self, id, capacity, occupancy=0, filled=False):
        self.id = id
        self.capacity = capacity
        self.occupancy = occupancy
        self.filled = filled

