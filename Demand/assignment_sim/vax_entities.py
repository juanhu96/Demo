
class Individual:
    def __init__(self, epsilon_diff):
        self.epsilon_diff = epsilon_diff # epsilon0 - epsilon1
        self.nlocs_considered = -1  # number of locations in consideration set
        self.location_assigned = -1  # location id of the location the individual is assigned to
        self.rank_assigned = -1  # rank of the location the individual is assigned to

class Geog:
    def __init__(self, location_ids, distances, abd, individuals):
        self.location_ids = location_ids
        self.distances = distances
        self.abd = abd
        self.individuals = individuals
        self.ab_epsilon = None # ab_epsilon[ll] = abd + distcoef * dist[ll]

# class Location:
#     def __init__(self, id, capacity, occupancy=0):
#         self.id = id
#         self.capacity = capacity
#         self.occupancy = occupancy
