#=================================================================
# NOTE
# DO NOT RUN THIS FILE, NEVER
# THIS IS JUST A TUTORIAL ON THE COMMAND

# CREATE AN TEMP SHELL FILE FOR ACTUAL RUN
#=================================================================

# the corresponding initializaiton is
nohup python3 initialization.py Dollar 10000 5 4 mnl flex logdistabove1.0 > output_log/init_10000_1_4q_mnl_flexthresh2_3_15_logdistabove1.0.txt
nohup python3 initialization.py Dollar 10000 5 4 mnl flex > output_log/init_10000_1_4q_mnl_flexthresh2_3_15.txt


# Baseline: 
# - Pharmacy-only
# - FacilityLocation, MaxVaxLogLin, 5 closest (only need to run once)

# Optimization, fix rank of 5
nohup python main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize > output_log/LogLin_optimize.txt
nohup python main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/LogLin_optimize.txt


# Evaluation
# when flexible threshold, max rank = 1, but M = 5 to ensure z is imported correctly
nohup python main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl flex logdistabove1.0 > output_log/LogLin_evaluate.txt


#=================================================================

# BLP
nohup python main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/BLP_optimize.txt
nohup python main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl flex logdistabove1.0 > output_log/BLP_evaluate.txt

#=================================================================

# Assortment requires full specification
nohup python main.py MNL Dollar 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/MNL_optimize.txt & # infeasibile in 8 hours
nohup python main.py MNL_partial Dollar 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/MNL_partial_optimize.txt & 

# Evaluate
nohup python main.py MNL_partial Dollar 10000 5 4 evaluate mnl flex logdistabove1.0 > output_log/MNL_partial_evaluate.txt

#=================================================================

# Summary results
nohup python summary.py 10000 5 4 mnl flex logdistabove1.0 > output_log/summary.txt