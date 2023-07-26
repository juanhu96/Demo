

import numpy as np
import pandas as pd
import pyblp
from typing import List, Optional, Tuple

try:
    from demand_utils.vax_entities import Economy, Individual
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils.vax_entities import Economy, Individual
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de

def assignment_difference(a1: List[np.ndarray], a2: List[np.ndarray]) -> float:
    """
    Compute the absolute difference in assignments for each location in each geography
    """
    total_diff = 0
    total_tallies = 0

    for geo_a1, geo_a2 in zip(a1, a2):
        # Make sure the two geographies have the same length by appending zeros to the shorter one
        if len(geo_a1) < len(geo_a2):
            geo_a1 = np.append(geo_a1, np.zeros(len(geo_a2) - len(geo_a1)))
        elif len(geo_a2) < len(geo_a1):
            geo_a2 = np.append(geo_a2, np.zeros(len(geo_a1) - len(geo_a2)))

        # Add the absolute differences in tallies for this geography to the total
        total_diff += np.sum(np.abs(geo_a1 - geo_a2))
        total_tallies += np.sum(geo_a1)

    # Return the fraction of tallies that are different
    return total_diff / total_tallies






def run_fp(
        economy:Economy,
        abd_init:np.ndarray,
        distcoefs_init:np.ndarray,
        locids:np.ndarray,
        capacity:float,
        ordering:List[int],
        agent_data_full:pd.DataFrame, #geog-loc level. need: blkid, locid, logdist, market_ids, nodes, hpi_quantile{qq}, logdistXhpi_quantile{qq}
        cw_pop:pd.DataFrame, #market-geog level. need: market_ids, blkid, population
        problem_init:pyblp.Problem, 
        maxiter:int = 100,
        verbose:bool = True
        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Run fixed point algorithm. 
    Stop when assignments stop changing or maxiter is reached.
    """

    


