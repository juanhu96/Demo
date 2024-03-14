#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2024
@author: Jingyuan Hu
"""

import sys
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pandas as pd

#=================================================================
# SETTINGS

K = int(sys.argv[1])
M = int(sys.argv[2])
nsplits = int(sys.argv[3])

capcoef = any(['capcoef' in arg for arg in sys.argv])
mnl = any([arg == 'mnl' for arg in sys.argv])
logdist_above = any(['logdistabove' in arg for arg in sys.argv])
if logdist_above:
    logdist_above_arg = [arg for arg in sys.argv if 'logdistabove' in arg][0]
    logdist_above_thresh = float(logdist_above_arg.replace('logdistabove', ''))
else: logdist_above_thresh = 0

flexible_consideration = any(['flex' in arg for arg in sys.argv])
flex_thresh = dict(zip(["urban", "suburban", "rural"], [2,3,15]))

replace = any(['replace' in arg for arg in sys.argv])
if replace:
    replace_arg = [arg for arg in sys.argv if 'replace' in arg][0]
    R = int(replace_arg.replace('replace', ''))
else: R = None

add = any(['add' in arg for arg in sys.argv])
if add:
    add_arg = [arg for arg in sys.argv if 'add' in arg][0]
    A = int(add_arg.replace('add', ''))
else: A = None

random = any(['random' in arg for arg in sys.argv])
if random:
    random_arg = [arg for arg in sys.argv if 'random' in arg][0]
    random_seed = int(random_arg.replace('random', ''))
else: random_seed = None

norandomterm = any(['norandomterm' in arg for arg in sys.argv]) # for log linear intercept
loglintemp = any(['loglintemp' in arg for arg in sys.argv]) # for log linear dist replace

setting_tag = f'_{str(K)}_1_{nsplits}q' if flexible_consideration else f'_{str(K)}_{M}_{nsplits}q' 
setting_tag += '_capcoefs0' if capcoef else ''
setting_tag += "_mnl" if mnl else ""
setting_tag += "_flex" if flexible_consideration else ""
setting_tag += f"thresh{str(list(flex_thresh.values())).replace(', ', '_').replace('[', '').replace(']', '')}" if flexible_consideration else ""
setting_tag += f"_logdistabove{logdist_above_thresh}" if logdist_above else ""


### MERGE RESULTS FROM EVERY RANDOMIZED INSTANCE
def merge_files_randomization(setting_tag, A_list = range(100, 1100, 100), random_seed_list = [42, 13, 940, 457, 129], resultdir = '/export/storage_covidvaccine/Result/Sensitivity_results/Randomization/'):

    total_df = pd.DataFrame()

    for A in A_list:
        for random_seed in random_seed_list:
            file_name = f'Results{setting_tag}_A{A}_randomseed{random_seed}'
            current_df = pd.read_csv(f'{resultdir}{file_name}.csv')
            if random_seed == 42: current_df['A'] = A
            current_df['random_seed'] = random_seed
            total_df = pd.concat([total_df, current_df])

    total_df.to_csv(f'{resultdir}Results{setting_tag}_randomization_merged.csv', encoding='utf-8', index=False, header=True)


    ### TABLE OF VACCINATIONS UNDER EACH A & RANDOM SEED
    summary_df = total_df.groupby(['A', 'random_seed']).agg(Vaccination=('Vaccination', 'sum'),
                                                            Vaccination_HPI1=('Vaccination HPI1', 'sum'),
                                                            Vaccination_HPI2=('Vaccination HPI2', 'sum'),
                                                            Vaccination_HPI3=('Vaccination HPI3', 'sum'),
                                                            Vaccination_HPI4=('Vaccination HPI4', 'sum')).reset_index()
    summary_df.to_csv(f'{resultdir}Summary{setting_tag}_randomization_merged.csv', encoding='utf-8', index=False, header=True)


    ### TABLE OF AVERAGE VACCINATIONS UNDER EACH A
    random_runs = 4 # 42 excluded as it only got evaluated 2 times (rather than 3)
    summary_df = summary_df[summary_df['random_seed'] != 42].groupby(['A']).agg(Vaccination=('Vaccination', lambda x: x.sum() / random_runs),
                                                            Vaccination_HPI1=('Vaccination_HPI1', lambda x: x.sum() / random_runs),
                                                            Vaccination_HPI2=('Vaccination_HPI2', lambda x: x.sum() / random_runs),
                                                            Vaccination_HPI3=('Vaccination_HPI3', lambda x: x.sum() / random_runs),
                                                            Vaccination_HPI4=('Vaccination_HPI4', lambda x: x.sum() / random_runs)).reset_index()
    summary_df.to_csv(f'{resultdir}Final{setting_tag}_randomization_merged.csv', encoding='utf-8', index=False, header=True)
    
    return
# merge_files_randomization(setting_tag)



### MERGE RESUTLS FROM EVERY PARTNERSHIPS
def merge_files_partnerships():

    pass