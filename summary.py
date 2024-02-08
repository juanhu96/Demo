#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu
"""

import sys
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from utils.partnerships_summary import partnerships_summary

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

#=================================================================


def summary(K, M, nsplits, capcoef, flexible_consideration, replace, R, A, setting_tag):

    print(f'Start creating summary table for {setting_tag}...\n')

    # Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistLogLin', 'MNL_partial']
    Model_list = ['MaxVaxDistLogLin', 'MNL_partial']
    # Model_list = ['MaxVaxDistLogLin'] # norandomterm # loglintemp
    partnerships_summary(Model_list=Model_list,
                         Chain_list=['Dollar', 'HighSchools', 'Coffee'],
                         K=K,
                         M=M,
                         nsplits=nsplits,
                         capcoef=capcoef,
                         flexible_consideration=flexible_consideration,
                         R=R,
                         A=A,
                         setting_tag=setting_tag,
                         suffix='chain_leftover')

    print(f'Finished table for {setting_tag}!\n')
    return


if __name__ == "__main__":
    summary(K, M, nsplits, capcoef, flexible_consideration, replace, R, A, setting_tag)