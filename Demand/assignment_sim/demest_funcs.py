from vax_entities import Economy, Individual
import numpy as np
import pandas as pd
import pyblp
from typing import List, Optional, Tuple

def estimate_demand(
        economy:Economy,
        weights, #returned by assignment (randon_fcfs)
        product_data:pd.DataFrame, 
        agent_data:pd.DataFrame, #need: distances, market_ids, blkid
        problem:pyblp.Problem
        ):

    """
    Estimate demand using BLP.
    """
    # make product data
    shares = shares_assigned(economy, df) #TODO:
    product_data = product_data.assign(shares=shares)

        # market_ids
        # shares from shares_assigned
        # controls 
        # hpi_quantile
        # firm_ids = 1
        # prices = 0 


    # make agent data

        # weights from dists_assigned
        # logdist from dists_assigned
        # hpi_quantile
        # market_ids
        # nodes = 0
        # interact logdist and hpi_quantile

    # make IVs

    # make problem
        # formulations
        # integration/optimization

    # solve problem

    # agent utilities - make function out of  snippet in demest_blocks


def shares_assigned(
    weights:List[np.array],
    df:pd.DataFrame
    ) -> pd.DataFrame:
    """
    Compute shares of each ZIP code from the block-level assignment
    Input: 
        weights: returned by assignment (random_fcfs)
        DataFrame at the geog level with columns market_ids, blkid, population
    Output: DataFrame at the market-level with columns market_ids, shares
    """
    #TODO: need to figure out indexing
    df['assigned_pop'] = np.array([np.sum(ww) for ww in weights])
    df_g = df.groupby('market_ids').agg({'population': 'sum', 'assigned_pop': 'sum'}).reset_index()
    df_g['share'] = df_g['assigned_pop'] / df_g['population']
    return df_g[['market_ids', 'share']]





def dists_assigned(
        weights, 
        distdf:pd.DataFrame
        ) -> pd.DataFrame:
    """
    Input: Economy object, distdf is the output of geonear in long format
    Output: DataFrame at the geog level with columns blkid, distances
    """
    # TODO: verify indexing
    # TODO: weights?
    nlocs_assigned = [len(tt) for tt in weights] # number of locations that were assigned in each geog
    nlocs_assigned_df = pd.DataFrame({'blkid': np.unique(distdf.blkid), 'nlocs_assigned': nlocs_assigned})
    distdf_tosub = distdf.merge(nlocs_assigned_df, on='blkid', how='left')
    distdf_subset = distdf_tosub.groupby('blkid').apply(lambda x: x.head(x['nlocs_assigned'].iloc[0])).reset_index(drop=True)
    distdf_subset = distdf_subset.drop(columns=['nlocs_assigned'])
    return distdf_subset


        


