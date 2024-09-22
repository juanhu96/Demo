"""
Created on Aug, 2024
@author: Jingyuan Hu

datadir = "/export/storage_covidvaccine/Demo/Data"

#########################################################################################

CONNECT BLOCK, TRACT, ZIP FOR ENTIRE STATE

1. read_block.py: read in block coordinates and population, save as blk_coords_pop.csv

2. ziptract.py: read in ZIP and tract shapefiles, make crosswalk between ZIPs and tracts

3. block_cw.py: read blk_coords_pop.csv, make crosswalks between blocks and ZIPs and between blocks and tracts

#########################################################################################

COMPUTE DISTANCES BETWEEN BLOCKS AND CURRENT LOCATIONS

4. block_dist.py: find distance to nearest locations for each block (requires locations.csv)

5. dist_all.py: distance between each block and ALL current locations

#########################################################################################

PREPARATION FOR DEMAND ESTIMATION

6. prep_block.py: prepare and merge block data for demand estimation

7. prep_zip.py: this subsumes prep_demest.do and trace back to raw data (requires areas.csv)

#########################################################################################

DEMAND ESTIMATION

8. demest_assm.py: demand estimation with capacity constraints

nsplits = 4, represents the unique values of RC (e.g., HPI quartile) in the data

python3 Demand/demest_assm.py 10000 5 4 mnl


#########################################################################################

INITIALIZATION

9. initialization.py: compute distance matrix and BLP matrix

python3 utils/initialization.py 10000 5 4 mnl


10. main.py: run optimization and pick the best set of locations

python3 main.py 10000 5 4 optimize mnl


11. main.py: evaluate new set of locations, report area-level rates

python3 main.py 10000 5 4 evaluate mnl

"""
