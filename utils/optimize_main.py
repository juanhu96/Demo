#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import pandas as pd
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.optimize_model import optimize_rate_MNL, optimize_rate_MNL_partial, optimize_rate_MNL_partial_new
from utils.import_parameters import import_basics, import_estimation


def optimize_main(K: int,
                  M: int,
                  nsplits: int,
                  capcoef: bool,
                  mnl: bool,
                  flexible_consideration: bool,
                  flex_thresh: dict,
                  logdist_above: bool,
                  logdist_above_thresh: float,
                  R, 
                  A,
                  norandomterm: bool,
                  loglintemp: bool,
                  random_seed,
                  setting_tag: str,
                  resultdir='/export/storage_covidvaccine/Demo/Result'):

    def create_path(K, M, nsplits, resultdir):
        path = f'{resultdir}/M{str(M)}_K{str(K)}_{nsplits}q/'
        if not os.path.exists(path): os.mkdir(path)
        return path
    
    expdirpath = create_path(K, M, nsplits, resultdir)
    optimize_chain(K, M, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, loglintemp, setting_tag, expdirpath)

    return



def optimize_chain(K: int,
                   M: int,
                   nsplits: int,
                   capcoef: bool,
                   mnl: bool,
                   flexible_consideration: bool,
                   flex_thresh: dict,
                   logdist_above: bool,
                   logdist_above_thresh: float,
                   R, 
                   A,
                   norandomterm: bool,
                   loglintemp: bool,
                   setting_tag: str,
                   expdirpath: str,
                   scale_factor: int = 10000):

    print(f'Optimization results stored at {expdirpath}...\n')
    
    # ================================================================================

    (locations, Population, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)

    V_total = import_estimation('V', R, A, None, setting_tag)
    max_value = np.max(V_total[np.isfinite(V_total)]) # replace np.inf w/ max value
    V_total[np.isinf(V_total)] = max_value

    v_total = V_total.flatten()
    pv_total = p_total * v_total
    v_total = v_total * Closest_total
    pv_total = pv_total * Closest_total # make sure zero at other place

    V_temp = v_total.reshape((num_tracts, num_total_stores))
    V_nonzero = np.where(V_temp == 0, np.inf, V_temp)
    v_min = np.min(V_nonzero, axis=1)
    big_M = np.where(v_min != 0, 1/v_min, 0)
    print(f'The min for correponding big M is {np.min(big_M)} and max is {np.max(big_M)}\n')
    gamma = (v_total * v_total) / (1 + v_total)
    pg_total = p_total * gamma
    pg_total = pg_total * Closest_total

    # ================================================================================

    path = expdirpath + '/'
    if not os.path.exists(path): os.mkdir(path)

    z_soln = optimize_rate_MNL_partial_new(scenario='total', 
                                           pg=pg_total,
                                           v=v_total,
                                           C=C,
                                           closest=Closest_total,
                                           K=K,
                                           big_M=big_M,
                                           R=R,
                                           A=A,
                                           num_current_stores=num_current_stores,
                                           num_total_stores=num_total_stores,
                                           num_tracts=num_tracts,
                                           path=path,
                                           setting_tag=setting_tag,
                                           scale_factor=scale_factor,
                                           MIPGap=1e-2)


    locations['selected'] = z_soln
    locations['selected'] = locations['selected'].astype(int)
    locations.to_csv(f"{expdirpath}/locations_output{setting_tag}.csv", index=False)

    print(f"Optimization finished. Locations output saved at {expdirpath}/locations_output{setting_tag}.csv\n")
    
    return 