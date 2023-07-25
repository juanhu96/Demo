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
    shares = shares_from_assignment(economy, df)
    product_data = product_data.assign(shares=shares)

    # make agent data
    agent_data = agent_data.assign(distances=distances, weights=weights)


    # make problem
    

    # solve problem

    # return results


def shares_from_assignment(economy:Economy, df:pd.DataFrame) -> np.ndarray: