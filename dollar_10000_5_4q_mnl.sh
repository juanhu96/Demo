# nohup python3 initialization.py Dollar 10000 5 4 mnl > output_log/init_10000_5_4q_mnl.txt 

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl > output_log/LogLin_opt_10000_5_4q_mnl.txt &
# nohup python3 main.py MNL Dollar 10000 5 4 optimize mnl > output_log/MNL_opt_10000_5_4q_mnl.txt & # unable to find a solution, weird
# nohup python3 main.py MNL Dollar 10000 5 4 optimize mnl > output_log/MNL_opt_10000_5_4q_mnl_lessequal.txt # able to find optimal
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl > output_log/MNL_partial_opt_10000_5_4q_mnl.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl > output_log/LogLin_eval_10000_5_4q_mnl.txt &
# nohup python3 main.py MNL Dollar 10000 5 4 evaluate mnl > output_log/MNL_eval_10000_5_4q_mnl_lessequal.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl > output_log/MNL_partial_eval_10000_5_4q_mnl.txt &

# nohup python3 summary.py 10000 5 4 mnl > output_log/summary_10000_5_4q_mnl.txt & 

# ============================================================
# Various R

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl 'replace100' > output_log/LogLin_opt_10000_5_4q_mnl_R100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'replace100' > output_log/MNL_partial_opt_10000_5_4q_mnl_R100.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl 'replace400' > output_log/LogLin_opt_10000_5_4q_mnl_R400.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'replace400' > output_log/MNL_partial_opt_10000_5_4q_mnl_R400.txt &






