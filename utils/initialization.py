#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct, 2023
@author: Jingyuan Hu
"""

import sys 
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

try:
    from initial_BLP import initial_BLP_estimation
except:
    from utils.initial_BLP import initial_BLP_estimation

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

setting_tag = f'_{str(K)}_1_{nsplits}q' if flexible_consideration else f'_{str(K)}_{M}_{nsplits}q' 
setting_tag += '_capcoefs0' if capcoef else ''
setting_tag += "_mnl" if mnl else ""
setting_tag += "_flex" if flexible_consideration else ""
setting_tag += f"thresh{str(list(flex_thresh.values())).replace(', ', '_').replace('[', '').replace(']', '')}" if flexible_consideration else ""
setting_tag += f"_logdistabove{logdist_above_thresh}" if logdist_above else ""

#=================================================================



def initialization(K, nsplits, capcoef, mnl, flexible_consideration, logdist_above, logdist_above_thresh, setting_tag):

    print(setting_tag)    
    initial_BLP_estimation(K, nsplits, capcoef, mnl, flexible_consideration, logdist_above, logdist_above_thresh, setting_tag)

    return



if __name__ == "__main__":
    initialization(K, nsplits, capcoef, mnl, flexible_consideration, logdist_above, logdist_above_thresh, setting_tag)