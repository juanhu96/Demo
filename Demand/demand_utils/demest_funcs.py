# mostly helper functions for estimating demand using pyblp

import numpy as np
import pandas as pd
import pyblp
from typing import List, Optional, Tuple



def hpi_dist_terms(
        df:pd.DataFrame,
        hpi_varname:str = 'hpi',
        dist_varname:str = 'logdist',
        nsplits:int = 4, #number of HPI quantiles
        add_bins:bool = False, #add hpi_quantile
        add_dummies:bool = False, #add hpi_quantile{qq}
        add_dist:bool = False #add logdistXhpi_quantile{qq}
        ) -> pd.DataFrame:

    # bin hpi into quantiles
    if add_bins:
        splits = np.linspace(0,1,nsplits+1)
        df = df.assign(hpi_quantile = pd.cut(df[hpi_varname], splits, labels=False, include_lowest=True) + 1)

    # add dummy variables for each quantile
    if add_dummies:
        for qq in range(1, nsplits+1):
            df[f'hpi_quantile{qq}'] = (df['hpi_quantile'] == qq).astype(int)

    # add the distance interaction term
    if add_dist:
        for qq in range(1, nsplits+1):
            df[f'{dist_varname}Xhpi_quantile{qq}'] = df[dist_varname] * df[f'hpi_quantile{qq}']

    return df



def add_ivcols(
        df:pd.DataFrame,
        agent_data:pd.DataFrame,
        agent_vars:List[str],
        verbose:bool = False
        ) -> pd.DataFrame:
    """
    Make instruments in product_data using the weighted average of agent variables
    """
    ivcols = []
    for (ii,vv) in enumerate(agent_vars):
        if verbose:
            print(f"demand_instruments{ii}: {vv}")
        ivcol = pd.DataFrame({f'demand_instruments{ii}': agent_data.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data.loc[x.index, 'weights']))})
        ivcols.append(ivcol)

    iv_df = pd.concat(ivcols, axis=1)
    df = df.merge(iv_df, left_on='market_ids', right_index=True)
    return df


def estimate_demand(
        df:pd.DataFrame, #need: market_ids, controls (incl. hpi_quantile), firm_ids=1, prices=0, shares
        agent_data:pd.DataFrame, #need: distances, market_ids, blkid, nodes
        product_formulations,
        agent_formulation,
        pi_init:np.ndarray = None,
        poolnum = 1,
        gtol = 1e-10,
        iteration_config = pyblp.Iteration(method='lm'),
        optimization_config = None,
        verbose:bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:

    """
    Simple wrapper for estimating demand using BLP - this is not very flexible and is meant to be run in the fixed point.
    """
    print("Estimating demand...")
    pyblp.options.verbose = verbose
    pyblp.options.flush_output = True
    
    agent_vars = str(agent_formulation).split(' + ')

    df = add_ivcols(df, agent_data, agent_vars=agent_vars, verbose=verbose)
    
    # set up and run BLP problem
    problem = pyblp.Problem(
        product_formulations=product_formulations, 
        product_data=df, 
        agent_formulation=agent_formulation, 
        agent_data=agent_data)
    

    if optimization_config is None:
        optimization_config = pyblp.Optimization('trust-constr', {'gtol':gtol})

    if pi_init is None:
        pi_init = 0.01*np.ones((1,len(agent_vars)))

    if poolnum==1:
        print("Solving with one core...")
        results = problem.solve(
            pi=pi_init,
            sigma = 0, 
            iteration = iteration_config, 
            optimization = optimization_config)
    else:
        with pyblp.parallel(poolnum): 
            results = problem.solve(
                pi=pi_init,
                sigma = 0, 
                iteration = iteration_config, 
                optimization = optimization_config)

    
    return results



def compute_abd(
        results:pyblp.ProblemResults,
        df:pd.DataFrame,
        agent_data:pd.DataFrame, #agent-level data with hpi_quantile{qq}, market_ids, and anything in pi_labels
        coefs = None, #optional. if not provided, use results.pi
        verbose:bool = False
) -> pd.DataFrame:
    
    """
    Compute agent-level utilities (all-but-distance) from pyblp results.
    Accounts for distance terms in the pi vector in the following form: logdist or logdistXhpi_quantile{qq}
    Requires agent_data to have columns blkid, market_ids, hpi_quantile{qq}, logdist.
    Output DataFrame has everything in agent_data plus agent_utility, distcoef, delta, abd

    """
    coefs = results.pi if coefs is None else coefs
    deltas = results.compute_delta(market_id = df['market_ids'])
    deltas_df = pd.DataFrame({'market_ids': df['market_ids'], 'delta': deltas.flatten()})
    if verbose:
        print(f"pi labels: {results.pi_labels}")

    agent_utils = agent_data.assign(
        agent_utility = 0,
        distcoef = 0
    )

    for (ii,vv) in enumerate(results.pi_labels):
        if verbose:
            print(f"ii={ii}, vv={vv}")
        coef = coefs[ii]
        if 'dist' in vv:
            if verbose:
                print(f"{vv} is a distance term, omitting from ABD and adding to coefficients instead")
            if vv=='logdist':
                agent_utils.loc[:, 'distcoef'] += coef
            elif vv.startswith('logdistXhpi_quantile'):
                qq = int(vv[-1]) #last character is the quantile number
                agent_utils.loc[:, 'distcoef'] +=  agent_data[f"hpi_quantile{qq}"] * coef
        else:
            if verbose:
                print(f"Adding {vv} to agent-level utility")
            agent_utils.loc[:, 'agent_utility'] += agent_data[vv] * coef 
            #agent_utility will be agent-level utility without distance terms. if all the agent-level terms are distance, we'll just be returning delta

    agent_utils = agent_utils.merge(deltas_df, on='market_ids')
    agent_utils = agent_utils.assign(abd = agent_utils['agent_utility'] + agent_utils['delta'])
    agent_utils.sort_values(by=['blkid', 'market_ids'], inplace=True)
    return agent_utils


from typing import List, Tuple

def start_table(tablevars: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Start table column with variable names
    """
    coefrows = []
    serows = []
    varlabels = []

    # Dictionary to map variable names to formatted labels
    format_dict = {
        '1': 'Constant',
        'logdist': 'Log(Distance)',
        'logdistXhpi_quantile1': 'Log(Distance) * HPI Quantile 1',
        'logdistXhpi_quantile2': 'Log(Distance) * HPI Quantile 2',
        'logdistXhpi_quantile3': 'Log(Distance) * HPI Quantile 3',
        'logdistXhpi_quantile4': 'Log(Distance) * HPI Quantile 4',
        'hpi_quantile1': 'HPI Quantile 1',
        'hpi_quantile2': 'HPI Quantile 2',
        'hpi_quantile3': 'HPI Quantile 3',
        'race_black': 'Race Black',
        'race_asian': 'Race Asian',
        'race_hispanic': 'Race Hispanic',
        'race_other': 'Race Other',
        'health_employer': 'Health Employer',
        'health_medicare': 'Health Medicare',
        'health_medicaid': 'Health Medicaid',
        'health_other': 'Health Other',
        'collegegrad': 'College Grad',
        'unemployment': 'Unemployment',
        'poverty': 'Poverty',
        'medianhhincome': 'Median Household Income',
        'logmedianhhincome': 'Log(Median Household Income)',
        'medianhomevalue': 'Median Home Value',
        'logmedianhomevalue': 'Log(Median Home Value)',
        'popdensity': 'Population Density',
        'logpopdensity': 'Log(Population Density)'
    }

    for vv in tablevars:
        vv_fmt = format_dict[vv]
        varlabels.append(vv_fmt)
        if 'dist' in vv:
            coefrows.append(f"{vv_fmt}" + "$^{\\dag}$ ")
        else:
            coefrows.append(f"{vv_fmt} ")
        serows.append(" ")

    return coefrows, serows, varlabels

def fill_table(
        results:pyblp.ProblemResults,
        coefrows:List[str],
        serows:List[str],
        tablevars:List[str]
        ) -> str:
    """
    Fill table column with coefficients and standard errors from pyblp results
    """

    betas = results.beta.flatten()
    betases = results.beta_se.flatten()
    betalabs = results.beta_labels
    pis = results.pi.flatten()
    pises = results.pi_se.flatten()
    pilabs = results.pi_labels

    for (ii,vv) in enumerate(tablevars):
        # print(f"ii={ii}, vv={vv}")
        if vv in betalabs:
            coef = betas[betalabs.index(vv)]
            coef_fmt = '{:.3f}'.format(coef)
            se = betases[betalabs.index(vv)]
            se_fmt = '(' + '{:.3f}'.format(se) + ')'
        elif vv in pilabs:
            coef = pis[pilabs.index(vv)] 
            coef_fmt = '{:.3f}'.format(coef)
            se = pises[pilabs.index(vv)] 
            se_fmt = '(' + '{:.3f}'.format(se) + ')'
        else: #empty cell if vv is not used in this config
            coef = 0
            coef_fmt = ''
            se = np.inf
            se_fmt = ''
        
        # add significance stars 
        if abs(coef/se) > 2.576:
            coef_fmt += '$^{***}$'
        elif abs(coef/se) > 1.96:
            coef_fmt += '$^{**}$'
        elif abs(coef/se) > 1.645:
            coef_fmt += '$^{*}$'

        # append to existing rows
        coefrows[ii] += f"& {coef_fmt}"
        serows[ii] += f"& {se_fmt}"

    return coefrows, serows
