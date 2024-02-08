# nohup python3 initialization.py Coffee 10000 5 4 mnl > output_log/Coffee/init_10000_5_4q_mnl.txt &

# nohup python3 main.py MNL_partial Coffee 10000 5 4 optimize mnl > output_log/Coffee/MNL_partial_opt_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Coffee 10000 5 4 optimize mnl > output_log/Coffee/LogLin_opt_10000_5_4q_mnl.txt &

# nohup python3 main.py MNL_partial Coffee 10000 5 4 evaluate mnl > output_log/Coffee/MNL_partial_eval_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Coffee 10000 5 4 evaluate mnl > output_log/Coffee/LogLin_eval_10000_5_4q_mnl.txt &

# LEFTOVER
nohup python3 main.py MNL_partial Coffee 10000 5 4 evaluate mnl > output_log/Coffee/MNL_partial_eval_leftover_10000_5_4q_mnl.txt &
nohup python3 main.py MaxVaxDistLogLin Coffee 10000 5 4 evaluate mnl > output_log/Coffee/LogLin_eval_leftover_10000_5_4q_mnl.txt &
