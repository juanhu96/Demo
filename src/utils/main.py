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
try:
    from utils.optimize_main import optimize_main
    from utils.evaluate_main import evaluate_main
except:
    from optimize_main import optimize_main
    from evaluate_main import evaluate_main

#=================================================================
# SETTINGS

K = int(sys.argv[1])
M = int(sys.argv[2])
nsplits = int(sys.argv[3])
stage = sys.argv[4]

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
setting_tag += f"_R{R}" if replace else ""
setting_tag += f"_A{A}" if add else ""
setting_tag += f"_norandomterm" if norandomterm else ""
setting_tag += f"_loglintemp" if loglintemp else ""
setting_tag += f"_randomseed{random_seed}" if random else ""

params = {
    'K': K,
    'M': M,
    'nsplits': nsplits,
    'capcoef': capcoef,
    'mnl': mnl, 
    'flexible_consideration': flexible_consideration, 
    'flex_thresh': flex_thresh, 
    'logdist_above': logdist_above, 
    'logdist_above_thresh': logdist_above_thresh, 
    'R': R, 
    'A': A,
    'norandomterm': norandomterm,
    'loglintemp': loglintemp,
    'random_seed': random_seed,
    'setting_tag': setting_tag
}


#=================================================================


def main(params):

    stage_functions = {
    'optimize': optimize_main,
    'evaluate': evaluate_main
    }

    if stage in stage_functions: stage_functions[stage](**params)
    else: raise Exception("Stage undefined\n")


if __name__ == "__main__":
    main(params)


