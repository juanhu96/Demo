# nohup python3 initialization.py Dollar 10000 5 4 mnl > output_log/init_10000_5_4q_mnl.txt 

nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl > output_log/LogLin_opt_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl > output_log/BLP_opt_10000_5_4q_mnl.txt &
nohup python3 main.py MNL Dollar 10000 5 4 optimize mnl > output_log/MNL_opt_10000_5_4q_mnl.txt &
nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl > output_log/MNL_partial_opt_10000_5_4q_mnl.txt &

