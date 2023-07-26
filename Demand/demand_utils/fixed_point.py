

import numpy as np
import pandas as pd
import pyblp
pyblp.options.digits = 3
from typing import List, Optional, Tuple

try:
    from demand_utils.vax_entities import Economy, Individual
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils.vax_entities import Economy, Individual
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de

def assignment_difference(a1: List[List[int]], a2: List[List[int]]) -> float:
    """
    Compute the difference in assignments
    """
    total_diff = 0
    total_tallies = 0

    for geo_a1, geo_a2 in zip(a1, a2):
        # Convert lists to np.array for easy manipulation
        geo_a1 = np.array(geo_a1)
        geo_a2 = np.array(geo_a2)
        
        # Make sure the two geographies have the same length by appending zeros to the shorter one
        if len(geo_a1) < len(geo_a2):
            geo_a1 = np.append(geo_a1, np.zeros(len(geo_a2) - len(geo_a1)))
        elif len(geo_a2) < len(geo_a1):
            geo_a2 = np.append(geo_a2, np.zeros(len(geo_a1) - len(geo_a2)))

        total_diff += np.sum(np.abs(geo_a1 - geo_a2))
        total_tallies += np.sum(geo_a1) + np.sum(geo_a2)

    # Return the fraction of tallies that are different
    return total_diff / total_tallies




def run_fp(
        economy:Economy,
        abd:np.ndarray,
        distcoefs:np.ndarray,
        capacity:float,
        agent_data_full:pd.DataFrame, #geog-loc level. need: blkid, locid, logdist, market_ids, nodes, hpi_quantile{qq}, logdistXhpi_quantile{qq}
        cw_pop:pd.DataFrame, #market-geog level. need: market_ids, blkid, population, 
        df:pd.DataFrame, #market-level data for product_data
        problem:pyblp.Problem, 
        gtol:float = 1e-10, #can change to 1e-8 when testing
        pi_init = None,
        maxiter:int = 100,
        tol = 1e-3
        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Run fixed point algorithm. 
    Stop when assignments stop changing or maxiter is reached.
    """
    adiff = 1 
    iter = 0

    while adiff > tol and iter < maxiter:
        a0 = economy.assignments

        # assignment
        af.compute_pref(economy, abd, distcoefs)
        af.random_fcfs(economy, capacity)
        if af.assignment_stats(economy) < 0.999:
            print("Warning: not all individuals are offered")
            return abd, distcoefs, df, agent_data
        
        a1 = economy.assignments
        adiff = assignment_difference(a0, a1)

        # demand estimation
        agent_data = af.subset_locs(economy.offers, agent_data_full)
        df = af.assignment_shares(df, economy.assignments, cw_pop)

    
        pi_init, agent_results = de.estimate_demand(df, agent_data, problem, pi_init=pi_init, gtol=gtol)
        abd = agent_results['abd'].values
        distcoefs = agent_results['distcoef'].values

        if np.any(distcoefs > 0):
            print(f"********\nWarning: distance coefficient is positive for some geographies in iteration {iter}.\n Replaced with -0.001")
            distcoefs[distcoefs > 0] = -0.001        
        print(f"\n************\nIteration {iter}\nAssignment difference: {adiff}")

        iter += 1

    return abd, distcoefs, df, agent_data
    


