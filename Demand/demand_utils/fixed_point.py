

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


def wdist_checker(a0, a1, dists_mm_sorted, sorted_indices, wdists, mm_where, tol=0.001):
    # break it up by market
    for (mm, mm_inds) in enumerate(mm_where):

        # mm_inds: indices that choose which blocks are in market mm
        a0_mm = [a0[i] for i in mm_inds]
        a1_mm = [a1[i] for i in mm_inds]
        d_mm = dists_mm_sorted[mm]

        p0 = cdf_assm(a0_mm, sorted_indices[mm]) # original distance distribution
        p1 = cdf_assm(a1_mm, sorted_indices[mm]) # new distance distribution
        
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
    # mm_where indices in cw_pop of each market's blocks
    mm_where = [np.where(cw_pop['market_ids'].values == mm)[0] for mm in mktinds]
    dists_mm = [np.concatenate([dists[tt] for tt in mm]) for mm in mm_where]
    sorted_indices = [np.argsort(dd) for dd in dists_mm]
    dists_mm_sorted = [dists_mm[i][sorted_indices[i]] for i in range(len(dists_mm))]
    wdists = np.zeros(len(mktinds)) #preallocate
    # dists_mm_sorted: list of arrays of distances. each array is a market, sorted by distance
    # sorted_indices: list of arrays of indices. each array is a market, sorted by distance
    return dists_mm_sorted, sorted_indices, wdists, mm_where



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
        tol = 0.001,
        dampener = 0, #dampener>0 to reduce step size
        coefsavepath:str = None,
        cap_coefs_to0:bool = False,
        mnl:bool = False,
        verbose:bool = False,
        setting_tag:str = None,
        outdir:str = None,
        strict_capacity:bool = False,
        dummy_location:bool = False,
        dummy_location_dist:float = None,
        pi_init:Optional[np.ndarray] = None
        ):
    """
    Run fixed point algorithm. 
    Stop when assignments stop changing or maxiter is reached.
    """
    pyblp.options.micro_computation_chunks = micro_computation_chunks
    if dampener > 0:
        print(f"Dampener: {dampener}")

    converged = False
    iter = 0
    dists_mm_sorted, sorted_indices, wdists, mm_where = wdist_init(cw_pop, economy.dists)
    agent_unique_data = agent_data_full.drop_duplicates(subset=['blkid']).copy()

    if verbose:
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        pd.options.display.width = None
    
        print("df:")
        print(df.head(50))

    while not converged and iter < maxiter:
        # subset agent_data to the offered locations
        offer_counts = np.concatenate(economy.offers) #initialized with everyone offered their nearest location
        assert len(offer_counts) == agent_data_full.shape[0]
        offer_inds = np.flatnonzero(offer_counts)
        offer_counts = offer_counts[offer_inds]
        print(f"Number of agents: {len(offer_inds)}")
        agent_loc_data = agent_data_full.loc[offer_inds].copy()
        agent_loc_data['weights'] = offer_counts / agent_loc_data['population']
        # agent_loc_data['population'] is the actual ZIP code population which works if we offer everyone.

        df['true_shares'] = df['shares']
        if strict_capacity: #we need to compute the shares as a fraction of the offered population
            agent_loc_data['frac_offered'] = agent_loc_data.groupby('market_ids')['weights'].transform('sum')
            agent_loc_data['weights'] = offer_counts / (agent_loc_data['frac_offered'] * agent_loc_data['population'])
            market_frac_offered = agent_loc_data[['market_ids', 'frac_offered']].groupby('market_ids', sort=True).first()
            df['frac_offered'] = df['market_ids'].map(market_frac_offered['frac_offered'])
            df['shares'] = df['true_shares'] / df['frac_offered']
            print("df:\n", df[['true_shares', 'shares', 'frac_offered']].head(50))
            print("agent_loc_data:\n", agent_loc_data[['weights', 'frac_offered', 'population']].head(50))
            if any(df['shares'] > 1):
                print(df.loc[df['shares'] > 1])
                print("Number of markets with shares > 1:", len(df.loc[df['shares'] > 1, 'market_ids']))
                # print out the agents for one of the markets 
                market_ids_sample = df.loc[df['shares'] > 1, 'market_ids'].iloc[0]
                print("agent_loc_data for one market that exceeds:\n", agent_loc_data.loc[agent_loc_data['market_ids'] == market_ids_sample])
                # Cap shares at 1
                df['shares'] = np.minimum(df['shares'], 0.99)
                # raise ValueError("Shares exceed 1")
            
        
        if dummy_location:
            agent_loc_data['offered'] = 1
            agent_loc_data['frac_offered'] = agent_loc_data.groupby('market_ids')['weights'].transform('sum')
            agent_loc_data['weights'] = offer_counts / (agent_loc_data['frac_offered'] * agent_loc_data['population'])

            violation_inds = np.flatnonzero(economy.agent_violations) # a count for each block
            if len(violation_inds) > 0:
                violation_weights = economy.agent_violations[violation_inds]
                print(f"Number of agents with violations: {len(violation_inds)}")
                violation_data = agent_data_full.loc[violation_inds].copy()
                violation_data[[cc for cc in agent_loc_data.columns if cc.startswith('logdist')]] = np.log(dummy_location_dist) #dummy location
                violation_data['weights'] = violation_weights / violation_data['population']
                violation_data['offered'] = 0
                agent_loc_data = pd.concat([agent_loc_data, violation_data], ignore_index=True)

        if iter == 0:
            if pi_init is not None:
                pi_init = pi_init.reshape((1,4))
        else:
            pi_init = results.pi

        if verbose:
            print(f"\nIteration {iter}:\n")
            print("agent_loc_data:")
            print(agent_loc_data.head(50))
            print("pi_init:")
            print(pi_init)
            print("Weight distribution by market:")
            print(agent_loc_data.groupby('market_ids')['weights'].sum().describe())
            sys.stdout.flush()

        results = de.estimate_demand(df, agent_loc_data, product_formulations, agent_formulation, pi_init=pi_init, gtol=gtol, poolnum=poolnum, verbose=verbose, dummy_location=dummy_location)

        # save table for first and last iterations
        if iter == 0:
            table_path = f"{outdir}/coeftables/coeftable_{setting_tag}_iter0.tex" 
            de.write_table(results, table_path)
        else:
            table_path = f"{outdir}/coeftables/coeftable_{setting_tag}.tex"


        coefs = results.pi.flatten() if iter==0 else coefs*dampener + results.pi.flatten()*(1-dampener)
        if cap_coefs_to0:
            coefs = np.minimum(coefs, 0)
        agent_results = de.compute_abd(results, df, agent_unique_data, coefs=coefs, verbose=verbose)

        abd = agent_results['abd'].values
        distcoefs = agent_results['distcoef'].values

        print(f"\nDistance coefficients: {[round(x, 5) for x in results.pi.flatten()]}\n")
        # save results
        if coefsavepath is not None:
            np.save(coefsavepath+str(iter), coefs)
            np.save(coefsavepath, coefs)
        
        if iter == 0:
            agent_results[['blkid', 'hpi_quantile', 'market_ids', 'abd', 'distcoef']].to_csv(f"{outdir}/agent_results_{setting_tag}_iter0.csv", index=False)
            results.to_pickle(f"{outdir}/results_{setting_tag}_iter0.pkl")
            agent_loc_data.to_csv(f"{outdir}/agent_loc_data_{setting_tag}_iter0.csv", index=False)
            print(f"Saved (agent_)results to {outdir}/(agent_)results_{setting_tag}_iter0.pkl")


        # assignment
        a0 = copy.deepcopy(economy.assignments)

        af.random_fcfs(economy, distcoefs, abd, capacity, mnl=mnl, strict_capacity=strict_capacity, dummy_location=dummy_location)

        af.assignment_stats(economy, max_rank=len(economy.offers[0]))
        converged = wdist_checker(a0, economy.assignments, dists_mm_sorted, sorted_indices, wdists, tol=tol, mm_where=mm_where)
        pd.DataFrame(economy.violation_count).to_csv(f"{outdir}/violation_count_{setting_tag}_iter{iter}.csv", index=False)
        economy.violation_count = [0]*len(economy.violation_count)
        print(f"Iteration {iter} complete.\n\n")
        sys.stdout.flush()
        iter += 1
    
    print(results)
    de.write_table(results, table_path)

    agent_results[['blkid', 'hpi_quantile', 'market_ids', 'abd', 'distcoef']].to_csv(f"{outdir}/agent_results_{setting_tag}.csv", index=False)
    results.to_pickle(f"{outdir}/results_{setting_tag}.pkl")
    agent_loc_data.to_csv(f"{outdir}/agent_loc_data_{setting_tag}.csv", index=False)
    print(f"Saved (agent_)results to {outdir}/(agent_)results_{setting_tag}.pkl")


    return agent_results, results, agent_loc_data
    
    