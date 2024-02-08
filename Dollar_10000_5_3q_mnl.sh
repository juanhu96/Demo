# nohup python3 initialization.py Dollar 10000 5 3 mnl > output_log/jan22/init_10000_5_3q_mnl.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 3 optimize mnl > output_log/jan22/MNL_partial_opt_10000_5_3q_mnl.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 3 optimize mnl > output_log/jan22/BLP_opt_10000_5_3q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 3 optimize mnl > output_log/jan22/LogLin_opt_10000_5_3q_mnl.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 3 evaluate mnl > output_log/jan22/MNL_partial_eval_10000_5_3q_mnl.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 3 evaluate mnl > output_log/jan22/BLP_eval_10000_5_3q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 3 evaluate mnl > output_log/jan22/LogLin_eval_10000_5_3q_mnl.txt &

# nohup python3 summary.py 10000 5 3 mnl > output_log/jan22/summary_10000_5_3q_mnl.txt & 


# LEFTOVER EVALUATION (Feb2)
nohup python3 main.py MNL_partial Dollar 10000 5 3 evaluate mnl > output_log/Feb2/MNL_partial_leftover_10000_5_3q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 3 evaluate mnl > output_log/Feb2/LogLin_partial_leftover_10000_5_3q_mnl.txt &

nohup python3 summary.py 10000 5 3 mnl > output_log/Feb2/summary_10000_5_3q_mnl.txt & 