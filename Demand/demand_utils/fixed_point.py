

import numpy as np
import pandas as pd
import pyblp
pyblp.options.digits = 3
from typing import List, Optional, Tuple
import copy

try:
    from demand_utils.vax_entities import Economy
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils.vax_entities import Economy
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de

def assignment_difference(a1: List[List[int]], a2: List[List[int]], total_pop: int) -> float:
    # Compute the difference in assignments
    total_diff = np.sum(np.abs(np.concatenate(a1) - np.concatenate(a2)))
    return total_diff / total_pop


def run_fp(
        economy:Economy,
        abd:np.ndarray,
        distcoefs:np.ndarray,
        capacity:float,
        agent_data_full:pd.DataFrame, #geog-loc level. need: blkid, locid, logdist, market_ids, nodes, hpi_quantile{qq}, logdistXhpi_quantile{qq}. need all the locs for each geog.
        cw_pop:pd.DataFrame, #market-geog level. need: market_ids, blkid, population. need to be sorted by blkid
        df:pd.DataFrame, #market-level data for product_data
        problem:pyblp.Problem, 
        gtol:float = 1e-10, #can change to 1e-8 when testing
        pi_init = None,
        maxiter:int = 100,
        tol = 1e-3,
        outdir:str = '/export/storage_covidvaccine/Data/Analysis/Demand'
        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Run fixed point algorithm. 
    Stop when assignments stop changing or maxiter is reached.
    """
    adiff = 1 
    iter = 0

    while adiff > tol and iter < maxiter:
        a0 = copy.deepcopy(economy.assignments)

        # assignment
        af.random_fcfs(economy, distcoefs, abd, capacity)
        if af.assignment_stats(economy) < 0.999:
            print("Warning: not all individuals are offered")
            return abd, distcoefs, df, agent_data
        
        adiff = assignment_difference(a0, economy.assignments, economy.total_pop)

        # demand estimation

        # subset agent_data to the locations that were actually offered
        offer_weights = np.concatenate(economy.offers)
        offer_inds = np.flatnonzero(offer_weights)
        agent_data = agent_data_full.loc[offer_inds]
        agent_data['weights'] = offer_weights[offer_inds]/agent_data['population']

        df = af.assignment_shares(df, economy.assignments, cw_pop)
        pi_iter = pi_init if iter == 0 else pi_result
        results, agent_results = de.estimate_demand(df, agent_data, problem, pi_init=pi_iter, gtol=gtol)
        pi_result = results.pi
        abd = agent_results['abd'].values
        distcoefs = agent_results['distcoef'].values

        print(f"\n************\nDistance coefficients: {pi_result}")
        if np.any(distcoefs > 0):
            print(f"\n************\nWarning: distance coefficient is positive for some geographies in iteration {iter}.")     
        print(f"\n************\nIteration {iter}\nAssignment difference: {adiff}")


        # save results
        results.to_pickle(f"{outdir}/pyblp_results_fp.pkl")
        agent_results.to_csv(f"{outdir}/agent_results_fp.csv")

        iter += 1

    return abd, distcoefs, df, agent_data
    
