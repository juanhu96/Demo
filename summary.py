#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul, 2022
@author: Jingyuan Hu
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys 
nsplits = int(sys.argv[1])
capcoef = sys.argv[2]
R = sys.argv[3]

from utils.partnerships_summary import partnerships_summary

def summary(nsplits, capcoef, R=None):

    '''
    BLP estimation:
    Capacity_list = [8000, 10000, 12000, 15000, 20000]

    Optimize, Evaluate, Partnership:
    Model_list = ['MaxVaxHPIDistBLP', 'MaxVaxDistBLP', 'MaxVaxHPIDistLogLin', 'MaxVaxDistLogLin', 'MaxVaxFixV', 'MinDist']
    Chain_list = ['Dollar', 'DiscountRetailers', 'Mcdonald', 'Coffee', 'ConvenienceStores', 'GasStations', 'CarDealers', 'PostOffices', 'HighSchools', 'Libraries']
    K_list = [8000, 10000, 12000]
    M_list = [5, 10]
    '''

    # partnerships_summary(Model_list=['MaxVaxHPIDistBLP', 'MaxVaxDistLogLin'], Chain_list=['Dollar'], M_list=[5], K_list=[8000, 10000], nsplits=nsplits, capcoef=capcoef, constraint_list=['vaccinated'], filename=str(nsplits))
    # partnerships_summary(Model_list=['MaxVaxHPIDistBLP'], Chain_list=['Dollar', 'Coffee', 'HighSchools'], M_list=[5], K_list=[8000], nsplits=nsplits, capcoef=capcoef, R=R, constraint_list=['vaccinated'])
    # partnerships_summary(Model_list=['MaxVaxHPIDistBLP', 'MaxVaxDistLogLin'], Chain_list=['Dollar'], M_list=[5], K_list=[8000], nsplits=nsplits, capcoef=capcoef, R=R, constraint_list=['vaccinated'], filename='BLPLogLin')
    partnerships_summary(Model_list=['MaxVaxHPIDistBLP', 'MaxVaxDistLogLin'], Chain_list=['Dollar'], M_list=[5], K_list=[8000], nsplits=nsplits, capcoef=capcoef, R=R, constraint_list=['vaccinated'], second_stage_MIP=True, filename='BLPLogLin')
    
    return


if __name__ == "__main__":
    
    if R != 'None':
        summary(nsplits, capcoef, int(R))
    else:
        summary(nsplits, capcoef)