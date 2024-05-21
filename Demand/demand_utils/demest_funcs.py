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
        add_hpi_bins:bool = False, #add hpi_quantile
        add_hpi_dummies:bool = False, #add hpi_quantile{qq}
        add_dist:bool = False, #add logdistXhpi_quantile{qq}
        add_distbins:bool = False, #add distbin{dd}Xhpi_quantile{qq}
        distbin_cuts:np.ndarray = [1,5], #distance quantile cuts
        logdist_above:bool = False, #if True, add logdist_abovethreshXhpi_quantile{qq} + abovethreshXhpi_quantile{qq}. Note that abovethreshXhpi_quantile{qq} might not be used in the model.
        logdist_above_thresh:float = 1.0 #threshold for logdist_above (this is in km not logkm)
        ) -> pd.DataFrame:

    # bin hpi into quantiles
    if add_hpi_bins:
        splits = np.linspace(0,1,nsplits+1)
        df = df.assign(hpi_quantile = pd.cut(df[hpi_varname], splits, labels=False, include_lowest=True) + 1)

    # add dummy variables for each quantile
    if add_hpi_dummies:
        for qq in range(1, nsplits+1):
            df[f'hpi_quantile{qq}'] = (df['hpi_quantile'] == qq).astype(int)

    # add the distance interaction term
    if add_dist:
        for qq in range(1, nsplits+1):
            df[f'{dist_varname}Xhpi_quantile{qq}'] = df[dist_varname] * df[f'hpi_quantile{qq}']

    # add the distance quantile interaction term
    if add_distbins:
        distbin_cuts_inf = np.concatenate(([-np.inf], distbin_cuts, [np.inf]))
        print("distbin_cuts_inf:", distbin_cuts_inf)
        df = df.assign(distbin = pd.cut(df[dist_varname], distbin_cuts_inf, labels=False, include_lowest=True))
        print("Distance quantiles:")
        print(df.distbin.value_counts())
        for qq in range(1, nsplits+1):
            # distance quantile interaction term starts at 1 (so 0 is the reference category)
            for dd in range(1, len(distbin_cuts_inf)-1):
                df[f'distbin{dd}Xhpi_quantile{qq}'] = (df['distbin'] == dd).astype(int) * df[f'hpi_quantile{qq}']
        print("Distance bin interactions:")
        print(df.filter(regex='distbin').sum())

    if logdist_above:
        for qq in range(1, nsplits+1):
            df[f'abovethreshXhpi_quantile{qq}'] = (df['dist'] > logdist_above_thresh).astype(int) * df[f'hpi_quantile{qq}']
            df[f'logdist_abovethreshXhpi_quantile{qq}'] = np.where(
                df[f'abovethreshXhpi_quantile{qq}'] == 1,
                np.log(np.maximum(df['dist'] + 1 - logdist_above_thresh, 1e-10)),
                0
            )
            
    return df



def add_ivcols(
        df:pd.DataFrame,
        agent_data:pd.DataFrame,
        agent_vars:List[str],
        dummy_location:bool = False,
        ) -> pd.DataFrame:
    """
    Make instruments in product_data using the weighted average of agent variables
    """
    if dummy_location:
        agent_data_for_iv = agent_data[agent_data['offered'] == 1]
    else:
        agent_data_for_iv = agent_data

    ivcols = []
    for (ii,vv) in enumerate(agent_vars):
        ivcol = pd.DataFrame({f'demand_instruments{ii}': agent_data_for_iv.groupby('market_ids')[vv].apply(lambda x: np.average(x, weights=agent_data_for_iv.loc[x.index, 'weights']))})
        ivcols.append(ivcol)

    if dummy_location: #insert share_not_offered
        currcol = 'demand_instruments' + str(len(agent_vars))
        iv_col = pd.DataFrame({currcol: 1 - agent_data.groupby('market_ids')['offered'].mean()})
        print("Share not offered:")
        print(iv_col[currcol].describe())
        # if there are any markets with people not offered
        if iv_col[currcol].max() > 0:
            ivcols.append(iv_col)

    iv_df = pd.concat(ivcols, axis=1)
    print("iv_df:\n", iv_df.head())
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
        verbose:bool = True,
        dummy_location:bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame]:

    """
    Simple wrapper for estimating demand using BLP - this is not very flexible and is meant to be run in the fixed point.
    """
    print("Estimating demand...")
    pyblp.options.verbose = verbose
    pyblp.options.flush_output = True
    
    agent_vars = str(agent_formulation).split(' + ')
    print(f"agent_vars: {agent_vars}")

    df = add_ivcols(df, agent_data, agent_vars=agent_vars, dummy_location=dummy_location)
    
    # set up and run BLP problem
    problem = pyblp.Problem(
        product_formulations=product_formulations, 
        product_data=df, 
        agent_formulation=agent_formulation, 
        agent_data=agent_data)
    

    if optimization_config is None:
        optimization_config = pyblp.Optimization('trust-constr', {'gtol':gtol})

    if pi_init is None:
        pi_init = -0.001*np.ones((1,len(agent_vars)))

    pi_ub = np.inf*np.ones(pi_init.shape)
    pi_lb = -np.inf*np.ones(pi_init.shape)

    print(f"pi_init: {pi_init}")
    print(f"pi_lb: {pi_lb}")
    print(f"pi_ub: {pi_ub}")


    if poolnum==1:
        print("Solving with one core...")
        results = problem.solve(
            pi=pi_init,
            pi_bounds=(pi_lb, pi_ub),
            sigma = 0, 
            iteration = iteration_config, 
            optimization = optimization_config)
    else:
        with pyblp.parallel(poolnum): 
            results = problem.solve(
                pi=pi_init,
                pi_bounds=(pi_lb, pi_ub),
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
        print(f"Mean delta: {deltas_df['delta'].mean()}")
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
            agent_utils.loc[:, 'agent_utility'] += agent_data[vv] * coef 
            #agent_utility will be agent-level utility without distance terms. if all the agent-level terms are distance, we'll just be returning delta

    agent_utils = agent_utils.merge(deltas_df, on='market_ids')
    agent_utils = agent_utils.assign(abd = agent_utils['agent_utility'] + agent_utils['delta'])
    agent_utils.sort_values(by=['blkid', 'market_ids'], inplace=True)
    return agent_utils


from typing import List, Tuple

def write_table(results:pyblp.ProblemResults, table_path:str):
    tablevars = results.pi_labels + results.beta_labels
    tablevars = [v for v in tablevars if v != 'prices']
    print("Table variables:", tablevars)

    coefrows, serows, varlabels = start_table(tablevars)
    coefrows, serows = fill_table(results, coefrows, serows, tablevars)

    coefrows = [r + "\\\\ \n" for r in coefrows]
    serows = [r + "\\\\ \n\\addlinespace\n" for r in serows]

    latex = "\\begin{tabular}{lcccc}\n \\toprule\n\\midrule\n"
    for (ii,vv) in enumerate(varlabels):
        latex += coefrows[ii]
        latex += serows[ii]

    latex += "\\bottomrule\n\\end{tabular}"
    
    with open(table_path, "w") as f:
        f.write(latex)


    print(f"Saved table at: {table_path}")



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
        'distbin1Xhpi_quantile1': 'Distance 1-5 * HPI Quantile 1',
        'distbin1Xhpi_quantile2': 'Distance 1-5 * HPI Quantile 2',
        'distbin1Xhpi_quantile3': 'Distance 1-5 * HPI Quantile 3',
        'distbin1Xhpi_quantile4': 'Distance 1-5 * HPI Quantile 4',
        'distbin2Xhpi_quantile1': 'Distance 5+ * HPI Quantile 1',
        'distbin2Xhpi_quantile2': 'Distance 5+ * HPI Quantile 2',
        'distbin2Xhpi_quantile3': 'Distance 5+ * HPI Quantile 3',
        'distbin2Xhpi_quantile4': 'Distance 5+ * HPI Quantile 4',
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
        try:
            vv_fmt = format_dict[vv]
        except:
            vv_fmt = vv.replace('_', ' ').replace('X', ' X ')
        varlabels.append(vv_fmt)
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
