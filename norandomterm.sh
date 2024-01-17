# Log linear w/o random terms

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize flex norandomterm > output_log/LogLin_opt_10000_1_4q_flexthresh2_3_15_norandomterm.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl flex norandomterm > output_log/LogLin_opt_10000_1_4q_mnl_flexthresh2_3_15_norandomterm.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl flex logdistabove1.0 norandomterm > output_log/LogLin_opt_10000_1_4q_mnl_flexthresh2_3_15_logdistabove1.0_norandomterm.txt & 


nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate flex norandomterm > output_log/LogLin_eval_10000_1_4q_flexthresh2_3_15_norandomterm.txt & 
nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl flex norandomterm > output_log/LogLin_eval_10000_1_4q_mnl_flexthresh2_3_15_norandomterm.txt & 
nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl flex logdistabove1.0 norandomterm > output_log/LogLin_eval_10000_1_4q_mnl_flexthresh2_3_15_logdistabove1.0_norandomterm.txt & 
