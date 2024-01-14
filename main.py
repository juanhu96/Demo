#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2023
@author: Jingyuan Hu
"""

import sys 
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from utils.optimize_main import optimize_main
from utils.evaluate_main import evaluate_main

#=================================================================
# SETTINGS

Model = sys.argv[1] # Model
Chain = sys.argv[2]
K = int(sys.argv[3])
M = int(sys.argv[4])
nsplits = int(sys.argv[5])
stage = sys.argv[6]

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

setting_tag = f'_{str(K)}_1_{nsplits}q' if flexible_consideration else f'_{str(K)}_{M}_{nsplits}q' 
setting_tag += '_capcoefs0' if capcoef else ''
setting_tag += "_mnl" if mnl else ""
setting_tag += "_flex" if flexible_consideration else ""
setting_tag += f"thresh{str(list(flex_thresh.values())).replace(', ', '_').replace('[', '').replace(']', '')}" if flexible_consideration else ""
setting_tag += f"_logdistabove{logdist_above_thresh}" if logdist_above else ""
setting_tag += f"_R{R}" if replace else ""

#=================================================================


def main(Model, Chain, K, M, nsplits, stage, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, replace, R, setting_tag):

    '''
    BLP estimation:
    Capacity_list = [8000, 10000, 12000, 15000, 20000]

    Optimize, Evaluate, Partnership:
    Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist']
    Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries']
    K_list = [8000, 10000, 12000]
    M_list = [5, 10]
    '''

    print(setting_tag)

    if stage == 'optimize': optimize_main(Model, Chain, K, M, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, replace, R, setting_tag)
    elif stage == 'evaluate': 
        evaluate_main(Model, Chain, K, M, nsplits, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, replace, R, setting_tag)
    else: raise Exception("Stage undefined\n")
    


if __name__ == "__main__":
    main(Model, Chain, K, M, nsplits, stage, capcoef, mnl, flexible_consideration, flex_thresh, logdist_above, logdist_above_thresh, replace, R, setting_tag)


