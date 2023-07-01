#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 21, 2022
@author: Jingyuan Hu
"""

import os
os.chdir('/export/storage_covidvaccine/Code')
from utils.optimize_main import optimize_main


def main():

    optimize_main(Model_list = ['MaxRateHPIDist', 'MaxRateDist'], Chain_list = ['Dollar'], Demand_estimation = 'linear')


if __name__ == "__main__":
    main()