#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@Author: Jingyuan Hu 
"""

import os
import numpy as np
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from utils.evaluate_model import compute_distdf, construct_blocks, run_assignment, summary_statistics
from utils.import_parameters import import_basics, import_estimation


def evaluate_main(K: int,
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
       
    path = create_path(K, M, nsplits, resultdir)
    evaluate_chain_RandomFCFS(M, K, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, R, A, norandomterm, random_seed, setting_tag, path)

    return



def evaluate_chain_RandomFCFS(M,
                              K,
                              nsplits,
                              capcoef,
                              mnl,
                              flexible_consideration,
                              flex_thresh,
                              logdist_above,
                              logdist_above_thresh,
                              R,
                              A,
                              norandomterm,
                              random_seed,
                              setting_tag,
                              path,
                              scale_factor: int = 10000):

    print(f'Evaluating random order FCFS; M = {str(M)}, K = {str(K)}, R = {R}, A = {A}.\n Results stored at {path}\n')

    (locations, Population, p_current, p_total, pc_current, pc_total, 
    C_total, Closest_current, Closest_total, _, _, C, num_tracts, 
    num_current_stores, num_total_stores) = import_basics(M, nsplits, flexible_consideration, logdist_above, logdist_above_thresh, scale_factor)

    z_file_name = f'{path}/z_total'
    z_total = np.genfromtxt(f'{z_file_name}{setting_tag}.csv', delimiter = ",", dtype = float)        
    print(f"Import optimization solution from file {z_file_name}{setting_tag}\n")

    # import pandas as pd
    # locations = pd.read_csv("/export/storage_covidvaccine/Demo/locations.csv")
    # z_total = locations['open'].values
    
    non_binary_check = np.any((z_total != 0) & (z_total != 1))
    print(f'Is there any fractional number in the final solution z: {non_binary_check}')
    if non_binary_check:
        print('The fractional numbers are')
        non_binary_indices = np.where((z_total != 0) & (z_total != 1))
        non_binary_values = z_total[non_binary_indices]
        for index, value in zip(non_binary_indices[0], non_binary_values):
            print(f"index: {index}, value: {value}")
        z_total = z_total.astype(int)
    
    compute_distdf(z_total, setting_tag, path)
    block, block_utils, distdf = construct_blocks(M, K, nsplits, flexible_consideration, flex_thresh, R, A, setting_tag, path)
    assignment, locs, dists = run_assignment(M, K, nsplits, capcoef, mnl, setting_tag, block, block_utils, distdf, path)

    summary_statistics(assignment, locs, dists, block, setting_tag, path)

    return