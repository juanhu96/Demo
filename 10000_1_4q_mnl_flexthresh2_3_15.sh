# MNL + Flexible Consideration + LogDist

# nohup python3 initialization.py Dollar 10000 5 4 mnl flex > output_log/init_10000_1_4q_mnl_flexthresh2_3_15.txt &

# wait

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl flex > output_log/LogLin_opt_10000_1_4q_mnl_flexthresh2_3_15.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl flex > output_log/BLP_opt_10000_1_4q_mnl_flexthresh2_3_15.txt & 
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl flex > output_log/MNL_partial_opt_10000_1_4q_mnl_flexthresh2_3_15.txt &

# wait

nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl flex > output_log/LogLin_eval_10000_1_4q_mnl_flexthresh2_3_15.txt & 
nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl flex > output_log/BLP_eval_10000_1_4q_mnl_flexthresh2_3_15.txt & 
nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl flex > output_log/MNL_partial_eval_10000_1_4q_mnl_flexthresh2_3_15.txt &

wait

nohup python3 summary.py 10000 5 4 mnl flex > output_log/summary_10000_1_4q_mnl_flexthresh2_3_15.txt & 