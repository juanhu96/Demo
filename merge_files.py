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
main = False
partnerships = False
randomization = False
parameters = False

#=================================================================
# SETTINGS

K = int(sys.argv[1])
M = int(sys.argv[2])
nsplits = int(sys.argv[3])
summary_case = sys.argv[4]

if summary_case == 'main': main = True
elif summary_case == 'partnerships': partnerships = True
elif summary_case == 'randomization': randomization = True
else: parameters = True

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
#=================================================================



### MERGE RESULTS FROM EVERY RANDOMIZED INSTANCE
def merge_files_randomization(setting_tag,
                              A_list = range(100, 1100, 100),
                              random_seed_list = [42, 13, 940, 457, 129],
                              export_details = False,
                              resultdir = '/export/storage_covidvaccine/Result/Sensitivity_results/Randomization/'):

    total_df = pd.DataFrame()
    
    ### IMPORT ALL INDIVIDUAL SUMMARY TABLE
    for A in A_list:
        for random_seed in random_seed_list:
            file_name = f'Results{setting_tag}_A{A}_randomseed{random_seed}'
            current_df = pd.read_csv(f'{resultdir}{file_name}.csv')
            # if random_seed == 42: current_df['A'] = A
            current_df['random_seed'] = random_seed
            total_df = pd.concat([total_df, current_df])
    if export_details: total_df.to_csv(f'{resultdir}Results{setting_tag}_randomization.csv', encoding='utf-8', index=False, header=True) # FCFS


    ### TABLE OF VACCINATIONS UNDER EACH A & RANDOM SEED
    summary_df = total_df.groupby(['A', 'random_seed']).agg(Vaccination=('Vaccination', 'sum'),
                                                            Vaccination_HPI1=('Vaccination HPI1', 'sum'),
                                                            Vaccination_HPI2=('Vaccination HPI2', 'sum'),
                                                            Vaccination_HPI3=('Vaccination HPI3', 'sum'),
                                                            Vaccination_HPI4=('Vaccination HPI4', 'sum')).reset_index()
    if export_details: summary_df.to_csv(f'{resultdir}Summary{setting_tag}_randomization.csv', encoding='utf-8', index=False, header=True) # FCFS


    ### TABLE OF AVERAGE VACCINATIONS UNDER EACH A
    random_runs = len(random_seed_list)
    summary_df = summary_df.groupby(['A']).agg(Vaccination=('Vaccination', lambda x: round(x.sum() / random_runs, 2)),
                                               Vaccination_HPI1=('Vaccination_HPI1', lambda x: round(x.sum() / random_runs, 2)),
                                               Vaccination_HPI2=('Vaccination_HPI2', lambda x: round(x.sum() / random_runs, 2)),
                                               Vaccination_HPI3=('Vaccination_HPI3', lambda x: round(x.sum() / random_runs, 2)),
                                               Vaccination_HPI4=('Vaccination_HPI4', lambda x: round(x.sum() / random_runs, 2))).reset_index()
    summary_df.to_csv(f'{resultdir}Final{setting_tag}_randomization.csv', encoding='utf-8', index=False, header=True)
    
    return


# ======================================================================



### MERGE RESUTLS FROM EVERY PARTNERSHIPS
def merge_files_partnerships(setting_tag, 
                             A_list = range(100, 1100, 100),
                            #  suffix = '_Dollar_FCFStest', # optimization vs. randomization
                             suffix = '_partnerships', # all chains
                             export_details = False,
                             resultdir = '/export/storage_covidvaccine/Result/Sensitivity_results/Partnerships/'):

    total_df = pd.DataFrame()
    
    ### IMPORT ALL INDIVIDUAL SUMMARY TABLE
    for A in A_list:
        file_name = f'Results{setting_tag}_A{A}{suffix}'
        current_df = pd.read_csv(f'{resultdir}{file_name}.csv')
        total_df = pd.concat([total_df, current_df])
    if export_details: total_df.to_csv(f'{resultdir}Results{setting_tag}_partnerships.csv', encoding='utf-8', index=False, header=True)
    

    ### TABLE OF VACCINATIONS UNDER EACH A & PARTNERSHIPS
    summary_df = total_df.groupby(['A', 'Chain']).agg(Vaccination=('Vaccination', lambda x: round(x.sum(), 2)),
                                                      Vaccination_HPI1=('Vaccination HPI1', lambda x: round(x.sum(), 2)),
                                                      Vaccination_HPI2=('Vaccination HPI2', lambda x: round(x.sum(), 2)),
                                                      Vaccination_HPI3=('Vaccination HPI3', lambda x: round(x.sum(), 2)),
                                                      Vaccination_HPI4=('Vaccination HPI4', lambda x: round(x.sum(), 2)),
                                                      Vaccination_Walkable =('Vaccination Walkable', lambda x: round(x.sum(), 2)),
                                                      Vaccination_Walkable_HPI1=('Vaccination Walkable HPI1', lambda x: round(x.sum(), 2)),
                                                      Vaccination_Walkable_HPI2=('Vaccination Walkable HPI2', lambda x: round(x.sum(), 2)),
                                                      Vaccination_Walkable_HPI3=('Vaccination Walkable HPI3', lambda x: round(x.sum(), 2)),
                                                      Vaccination_Walkable_HPI4=('Vaccination Walkable HPI4', lambda x: round(x.sum(), 2))).reset_index()
    
    summary_df.to_csv(f'{resultdir}Final{setting_tag}_partnerships.csv', encoding='utf-8', index=False, header=True)

    ### TABLE OF VACCINATION UNDER DIFF A (DOLLAR ONLY FOR COMPARISON)
    optimization_df = summary_df[summary_df['Chain'] == 'Pharmacy + Dollar']
    optimization_df = optimization_df.drop(columns=['Chain', 'Vaccination_Walkable',\
                                                    'Vaccination_Walkable_HPI1', 'Vaccination_Walkable_HPI2',\
                                                        'Vaccination_Walkable_HPI3', 'Vaccination_Walkable_HPI4'])
    optimization_df.to_csv(f'{resultdir}Final{setting_tag}_optimization.csv', encoding='utf-8', index=False, header=True)

    return



# ======================================================================



### MERGE RESUTLS FROM EVERY PARTNERSHIPS
def merge_files_parameters(suffix = '_parameter',
                           export_details = True,
                           resultdir = '/export/storage_covidvaccine/Result/Sensitivity_results/Parameters/'):

    total_df = pd.DataFrame()
    
    ### IMPORT ALL INDIVIDUAL SUMMARY TABLE
    # full replacement
    setting_tag_list_pharmacy = ['_10000_5_4q_mnl', '_10000_10_4q_mnl', '_8000_5_4q_mnl', '_12000_5_4q_mnl',
                        '_10000_5_4q_mnl_logdistabove0.8', '_10000_5_4q_mnl_logdistabove1.6']
    # add 500
    setting_tag_list_A500 = ['_10000_5_4q_mnl_A500', '_10000_10_4q_mnl_A500', '_8000_5_4q_mnl_A500', '_12000_5_4q_mnl_A500',
                        '_10000_5_4q_mnl_logdistabove0.8_A500', '_10000_5_4q_mnl_logdistabove1.6_A500']

    setting_tag_list = []
    for item1, item2 in zip(setting_tag_list_pharmacy, setting_tag_list_A500):
        setting_tag_list.append(item1)
        setting_tag_list.append(item2)

    for setting_tag in setting_tag_list:
        if setting_tag == '_10000_5_4q_mnl_logdistabove0.8': d = 0.5
        elif setting_tag == '_10000_5_4q_mnl_logdistabove1.6': d = 1.0
        else: d = 0

        file_name = f'Results{setting_tag}{suffix}'
        current_df = pd.read_csv(f'{resultdir}{file_name}.csv')
        current_df['d'] = d
        total_df = pd.concat([total_df, current_df])

    if export_details: total_df.to_csv(f'{resultdir}Results_parameters.csv', encoding='utf-8', index=False, header=True)
    
    columns_to_keep = ['M', 'K', 'd',
                       'Vaccination', 
                       'Vaccination HPI1', 'Vaccination HPI2',
                       'Vaccination HPI3', 'Vaccination HPI4',
                       'Vaccination Walkable', 
                       'Vaccination Walkable HPI1', 'Vaccination Walkable HPI2',
                       'Vaccination Walkable HPI3', 'Vaccination Walkable HPI4']
    
    columns_to_round = ['Vaccination', 'Vaccination HPI1', 'Vaccination HPI2', 'Vaccination HPI3', 'Vaccination HPI4',
                        'Vaccination Walkable', 'Vaccination Walkable HPI1', 'Vaccination Walkable HPI2',
                        'Vaccination Walkable HPI3', 'Vaccination Walkable HPI4']
    
    summary_df = total_df[columns_to_keep]
    summary_df[columns_to_round] = summary_df[columns_to_round].round(2)
    # print(summary_df)
    # summary_df = summary_df.iloc[::2].reset_index(drop=True) - summary_df.iloc[1::2].reset_index(drop=True)
    summary_df = summary_df.iloc[1::2].reset_index(drop=True) - summary_df.iloc[::2].reset_index(drop=True)
    vaccination_df = summary_df[['M', 'K', 'd', 'Vaccination', 'Vaccination HPI1', 'Vaccination HPI2', 'Vaccination HPI3', 'Vaccination HPI4']]
    vaccination_df['Type'] = 'Total'

    walkable_df = summary_df[['M', 'K', 'd', 'Vaccination Walkable', 'Vaccination Walkable HPI1', 'Vaccination Walkable HPI2', 'Vaccination Walkable HPI3', 'Vaccination Walkable HPI4']]
    new_column_names = {
    'Vaccination Walkable': 'Vaccination',
    'Vaccination Walkable HPI1': 'Vaccination HPI1',
    'Vaccination Walkable HPI2': 'Vaccination HPI2',
    'Vaccination Walkable HPI3': 'Vaccination HPI3',
    'Vaccination Walkable HPI4': 'Vaccination HPI4'
    }
    walkable_df.rename(columns=new_column_names, inplace=True)
    walkable_df['Type'] = 'Walkable'

    combined_df = pd.concat([vaccination_df, walkable_df], ignore_index=True)
    interleave_indices = [val for pair in zip(range(len(vaccination_df)), range(len(vaccination_df), len(combined_df))) for val in pair]
    interleaved_df = combined_df.iloc[interleave_indices].reset_index(drop=True)
    interleaved_df = interleaved_df[['Type', 'Vaccination', 'Vaccination HPI1', 'Vaccination HPI2', 'Vaccination HPI3', 'Vaccination HPI4']]
    # print(interleaved_df)

    latex_table = interleaved_df.to_latex(index=False)
    print(latex_table)

    return


# ======================================================================



def merge_files_main(setting_tag,
                     suffix='',
                     resultdir='/export/storage_covidvaccine/Result/Sensitivity_results/'):

    file_name = f'Results{setting_tag}{suffix}'
    df = pd.read_csv(f'{resultdir}{file_name}.csv')
    
    summary_df = df.groupby(['Model', 'Chain']).agg(M=('M', 'first'),
                                           K=('K', 'first'),
                                           Pharmacies_replaced = ('Pharmacies replaced', 'first'),
                                           Pharmacies_replaced_HPI1 = ('Pharmacies replaced HPI 1', 'first'),
                                           Pharmacies_replaced_HPI2 = ('Pharmacies replaced HPI 2', 'first'),
                                           Pharmacies_replaced_HPI3 = ('Pharmacies replaced HPI 3', 'first'),
                                           Pharmacies_replaced_HPI4 = ('Pharmacies replaced HPI 4', 'first'),
                                           Stores_opened_HPI1 = ('Stores opened HPI 1', 'first'),
                                           Stores_opened_HPI2 = ('Stores opened HPI 2', 'first'),
                                           Stores_opened_HPI3 = ('Stores opened HPI 3', 'first'),
                                           Stores_opened_HPI4 = ('Stores opened HPI 4', 'first'),
                                           Vaccination=('Vaccination', lambda x: round(x.sum(), 2)),
                                           Vaccination_HPI1=('Vaccination HPI1', lambda x: round(x.sum(), 2)),
                                           Vaccination_HPI2=('Vaccination HPI2', lambda x: round(x.sum(), 2)),
                                           Vaccination_HPI3=('Vaccination HPI3', lambda x: round(x.sum(), 2)),
                                           Vaccination_HPI4=('Vaccination HPI4', lambda x: round(x.sum(), 2)),
                                           Vaccination_Walkable =('Vaccination Walkable', lambda x: round(x.sum(), 2)),
                                           Vaccination_Walkable_HPI1=('Vaccination Walkable HPI1', lambda x: round(x.sum(), 2)),
                                           Vaccination_Walkable_HPI2=('Vaccination Walkable HPI2', lambda x: round(x.sum(), 2)),
                                           Vaccination_Walkable_HPI3=('Vaccination Walkable HPI3', lambda x: round(x.sum(), 2)),
                                           Vaccination_Walkable_HPI4=('Vaccination Walkable HPI4', lambda x: round(x.sum(), 2))).reset_index()
    
    summary_df['Stores net gain HPI 1'] = summary_df['Stores_opened_HPI1'] - summary_df['Pharmacies_replaced_HPI1']
    summary_df['Stores net gain HPI 2'] = summary_df['Stores_opened_HPI2'] - summary_df['Pharmacies_replaced_HPI2']
    summary_df['Stores net gain HPI 3'] = summary_df['Stores_opened_HPI3'] - summary_df['Pharmacies_replaced_HPI3']
    summary_df['Stores net gain HPI 4'] = summary_df['Stores_opened_HPI4'] - summary_df['Pharmacies_replaced_HPI4']

    # columns_to_drop = ['Stores net gain HPI 1', 'Stores net gain HPI 2', 'Stores net gain HPI 3', 'Stores net gain HPI 4']
    # summary_df = summary_df.drop(columns=columns_to_drop)

    summary_df.to_csv(f'{resultdir}Final{setting_tag}_replacement{suffix}.csv', encoding='utf-8', index=False, header=True)

    return


if __name__ == "__main__":
    if main: merge_files_main(setting_tag)
    if randomization: merge_files_randomization(setting_tag)
    if partnerships: merge_files_partnerships(setting_tag)
    if parameters: merge_files_parameters()