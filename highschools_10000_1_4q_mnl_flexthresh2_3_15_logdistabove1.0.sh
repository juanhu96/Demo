# so

nohup python3 initialization.py HighSchools 10000 5 4 mnl flex logdistabove1.0 > output_log/highschools/init_10000_1_4q_mnl_flexthresh2_3_15_logdistabove1.0.txt &

wait

nohup python3 main.py MaxVaxDistLogLin HighSchools 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/highschools/LogLin_opt_10000_1_4q_mnl_flexthresh2_3_15.txt & 
nohup python3 main.py MaxVaxHPIDistBLP HighSchools 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/highschools/BLP_opt_10000_1_4q_mnl_flexthresh2_3_15.txt & 
nohup python3 main.py MNL_partial HighSchools 10000 5 4 optimize mnl flex logdistabove1.0 > output_log/highschools/MNL_partial_opt_10000_1_4q_mnl_flexthresh2_3_15.txt &

wait

nohup python3 main.py MaxVaxDistLogLin HighSchools 10000 5 4 evaluate mnl flex logdistabove1.0 > output_log/highschools/LogLin_eval_10000_1_4q_mnl_flexthresh2_3_15.txt & 
nohup python3 main.py MaxVaxHPIDistBLP HighSchools 10000 5 4 evaluate mnl flex logdistabove1.0 > output_log/highschools/BLP_eval_10000_1_4q_mnl_flexthresh2_3_15.txt & 
nohup python3 main.py MNL_partial HighSchools 10000 5 4 evaluate mnl flex logdistabove1.0 > output_log/highschools/MNL_partial_eval_10000_1_4q_mnl_flexthresh2_3_15.txt &