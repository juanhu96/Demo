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

    optimize_main(Model_list = ['MinDistNew'], Chain_list = ['Dollar'], K_list = [10000, 12000], Demand_estimation = 'BLP')


if __name__ == "__main__":
    main()