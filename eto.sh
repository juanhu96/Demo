# Created on Aug, 2024
# @author: Jingyuan Hu

# datadir = "/export/storage_covidvaccine/Demo/Data"

#########################################################################################

# CONNECT BLOCK, TRACT, ZIP FOR ENTIRE STATE

# 1. read_block.py: read in block coordinates and population, save as blk_coords_pop.csv

# 2. ziptract.py: read in ZIP and tract shapefiles, make crosswalk between ZIPs and tracts

# 3. block_cw.py: read blk_coords_pop.csv, make crosswalks between blocks and ZIPs and between blocks and tracts

#########################################################################################

# COMPUTE DISTANCES BETWEEN BLOCKS AND CURRENT LOCATIONS

# 4. block_dist.py: find distance to nearest locations for each block (requires locations.csv)

# 5. dist_all.py: distance between each block and ALL current locations

#########################################################################################

# PREPARATION FOR DEMAND ESTIMATION

# 6. prep_block.py: prepare and merge block data for demand estimation

# 7. prep_zip.py: this subsumes prep_demest.do and trace back to raw data (requires areas.csv)

#########################################################################################

# DEMAND ESTIMATION

# 8. demest_assm.py: demand estimation with capacity constraints

# nsplits = 4, represents the unique values of RC (e.g., HPI quartile) in the data

# python3 Demand/demest_assm.py 10000 5 4 mnl


#########################################################################################

# INITIALIZATION

# 9. initialization.py: compute distance matrix and BLP matrix

# python3 utils/initialization.py 10000 5 4 mnl


# 10. main.py: run optimization and pick the best set of locations

# python3 main.py 10000 5 4 optimize mnl


# 11. main.py: evaluate new set of locations, report area-level rates

# python3 main.py 10000 5 4 evaluate mnl

#########################################################################################

#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <value1> <value2>"
    echo "Example: $0 10000 5 where the first number (10000) is the capacity per location and the second number (5) is size of your choice set"
    exit 1
fi

value1=$1
value2=$2

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="log/output_log_${timestamp}.txt"


# python3 src/Demand/datawork/block/read_block.py >> "$output_file" 2>&1
# python3 src/Demand/datawork/ziptract.py >> "$output_file" 2>&1
# python3 src/Demand/datawork/block/block_cw.py >> "$output_file" 2>&1
# python3 src/Demand/datawork/block/block_dist.py >> "$output_file" 2>&1
# python3 src/Demand/datawork/dist_all.py >> "$output_file" 2>&1
# python3 src/Demand/datawork/block/prep_block.py >> "$output_file" 2>&1
# python3 src/Demand/datawork/prep_zip.py >> "$output_file" 2>&1
python3 src/Demand/demest_assm.py $value1 $value2 4 mnl >> "$output_file" 2>&1
# python3 src/utils/initialization.py $value1 $value2 4 mnl >> "$output_file" 2>&1
# python3 src/utils/main.py 10000 5 4 optimize mnl >> "$output_file" 2>&1
# python3 src/utils/main.py 10000 5 4 evaluate mnl >> "$output_file" 2>&1

echo "Model executed successfully! The selected location is exported as ... and the log file is saved as $output_file."
