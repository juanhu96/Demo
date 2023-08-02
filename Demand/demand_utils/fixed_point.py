

import numpy as np
import pandas as pd
import pyblp
pyblp.options.digits = 3
pyblp.options.flush_output = True
pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0 # prices=0 always triggers collinearity warning. there are no other warnings.
from typing import List, Optional, Tuple
import copy
from itertools import compress
import sys

try:
    from demand_utils.vax_entities import Economy
    from demand_utils import assignment_funcs as af
    from demand_utils import demest_funcs as de
except:
    from Demand.demand_utils.vax_entities import Economy
    from Demand.demand_utils import assignment_funcs as af
    from Demand.demand_utils import demest_funcs as de



def assignment_difference(original_assm: List[List[int]], economy:Economy, cw_pop:pd.DataFrame, tol:float) -> float:
    # Compute the fraction of individuals whose assignment changed
    total_diff = np.sum(np.abs(np.concatenate(original_assm) - np.concatenate(economy.assignments)))
    adiff = total_diff / economy.total_pop
    print(f"Assignment difference: {adiff}")
    return adiff < tol



def cdf_assm(assm: List[np.ndarray], sorted_indices: List[np.ndarray]):
    # compute the probability distribution of assignments on distances
    # helper function for wasserstein_distance
    assm_flat = np.concatenate(assm)
    assm_sorted = assm_flat[sorted_indices] # sort by distance
    total_assm = np.sum(assm_sorted)
    if total_assm == 0:
        return np.zeros_like(assm_sorted)
    probdist = assm_sorted / total_assm
    return probdist


def wasserstein_distance(p0: np.ndarray, p1: np.ndarray, dists: np.ndarray) -> float:
    # compute the Wasserstein distance between two probability distributions
    # just the weighted sum of absolute differences between the CDFs, where the weights are the distances
    cdf0 = np.cumsum(p0)
    cdf1 = np.cumsum(p1)
    dist_diffs = np.diff(dists)
    wasserstein_dist = np.sum(np.abs(cdf0 - cdf1)[:-1] * dist_diffs)

    return wasserstein_dist


def wdist_checker(a0, a1, dists_mm_sorted, sorted_indices, wdists, tol):
    # break it up by market
    for (mm, mm_inds) in enumerate(sorted_indices):
        d_mm = dists_mm_sorted[mm]
        original_assm_mm = [a0[i] for i in mm_inds]
        new_assm_mm = [a1[i] for i in mm_inds]
        p0 = cdf_assm(original_assm_mm, sorted_indices[mm]) # original distance distribution
        p1 = cdf_assm(new_assm_mm, sorted_indices[mm]) # new distance distribution
        wasserstein_dist = wasserstein_distance(p0, p1, d_mm)
        wdists[mm] = wasserstein_dist
    print(f"Number of markets with wasserstein distance > {tol}: {np.sum(wdists > tol)}")
    if np.sum(wdists > tol) > 0:
        print(f"Mean wasserstein distance for markets with wasserstein distance > {tol}: {np.mean(wdists[wdists > tol])}")
    print(f"Mean wasserstein distance: {np.mean(wdists)}")
    print(f"Max wasserstein distance: {np.max(wdists)}")
    print(f"Index of max wasserstein distance: {np.argmax(wdists)}")
    return np.sum(wdists > tol) == 0



def wdist_init(cw_pop:pd.DataFrame, dists:List[np.ndarray]):
    # one-time stuff to speed up wdist_checker
    cw_pop = cw_pop.reset_index(drop=True)
    cw_pop['geog_ind'] = cw_pop.index
    mktinds = np.unique(cw_pop['market_ids'])
    mm_where = [np.where(cw_pop['market_ids'].values == mm)[0] for mm in mktinds]
    dists_mm = [np.concatenate([dists[tt] for tt in mm]) for mm in mm_where]
    sorted_indices = [np.argsort(dd) for dd in dists_mm]
    dists_mm_sorted = [dists_mm[i][sorted_indices[i]] for i in range(len(dists_mm))]
    wdists = np.zeros(len(mktinds)) #preallocate

    return dists_mm_sorted, sorted_indices, wdists



def run_fp(
        economy:Economy,
        capacity:float,
        agent_data_full:pd.DataFrame, #geog-loc level. need: blkid, locid, logdist, market_ids, nodes, hpi_quantile{qq}, logdistXhpi_quantile{qq}. need all the locs for each geog.
        cw_pop:pd.DataFrame, #market-geog level. need: market_ids, blkid, population. need to be sorted by blkid
        df:pd.DataFrame, #market-level data for product_data
        product_formulations,
        agent_formulation,
        gtol:float = 1e-10, #can change to 1e-8 when testing
        poolnum:int = 1,
        micro_computation_chunks:Optional[int] = 1,
        maxiter:int = 100,
        tol = 0.002,
        outdir:str = '/export/storage_covidvaccine/Data/Analysis/Demand'
        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Run fixed point algorithm. 
    Stop when assignments stop changing or maxiter is reached.
    """
    pyblp.options.micro_computation_chunks = micro_computation_chunks

    converged = False
    iter = 0
    dists_mm_sorted, sorted_indices, wdists = wdist_init(cw_pop, economy.dists)


    while not converged and iter < maxiter:

        # subset agent_data to the offered locations
        offer_weights = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
        offer_inds = np.flatnonzero(offer_weights)
        offer_weights = offer_weights[offer_inds]
        print(f"Number of agents: {len(offer_inds)}")
        agent_data = agent_data_full.loc[offer_inds].copy()
        agent_data['weights'] = offer_weights/agent_data['population']

        pi_init = results.pi if iter > 0 else 0.001*np.ones((1, len(str(agent_formulation).split('+')))) #initialize pi to last result, unless first iteration
        results, agent_results = de.estimate_demand(df, agent_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=gtol, poolnum=poolnum, verbose=False)
        abd = agent_results['abd'].values
        distcoefs = agent_results['distcoef'].values

        print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")

        # assignment
        a0 = copy.deepcopy(economy.assignments)
        af.random_fcfs(economy, distcoefs, abd, capacity)
        af.assignment_stats(economy)
        converged = wdist_checker(a0, economy.assignments, dists_mm_sorted, sorted_indices, wdists, tol)
        print(f"Iteration {iter}")
        iter += 1

    print(f"\n************\nExiting fixed point.\nSaved results to {outdir}")
    return abd, distcoefs, df, agent_data
    
