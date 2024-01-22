# above 0.5, 1.0, 1.6

# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove0.5" > output_log/Demand_10000_5_4q_mnl_logdistabove0.5.txt
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove1.0" > output_log/Demand_10000_5_4q_mnl_logdistabove1.0.txt
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove1.6" > output_log/Demand_10000_5_4q_mnl_logdistabove1.6.txt

# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove0.5" > output_log/init_10000_5_4q_mnl_logdistabove0.5.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1.0" > output_log/init_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1.6" > output_log/init_10000_5_4q_mnl_logdistabove1.6.txt &

# ===========================================================================================

# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove0.5" > output_log/MNL_partial_opt_10000_5_4q_mnl_logdistabove0.5.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.0" > output_log/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.6" > output_log/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.6.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove0.5" > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove0.5.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.0" > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.6" > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove1.6.txt &

# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove0.5" > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove0.5.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.0" > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove1.0.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.6" > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove1.6.txt & 


# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove0.5" 'replace100' > output_log/new/MNL_partial_opt_10000_5_4q_mnl_logdistabove0.5_R100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'replace100' > output_log/new/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0_R100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.6" 'replace100' > output_log/new/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.6_R100.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove0.5" 'replace100' > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove0.5_R100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'replace100' > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove1.0_R100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.6" 'replace100' > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove1.6_R100.txt &

# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove0.5" 'replace100' > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove0.5_R100.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'replace100' > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove1.0_R100.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.6" 'replace100' > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove1.6_R100.txt & 


# ===========================================================================================
# ADD 100
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'add100' > output_log/new/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0_A100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'add100' > output_log/new/LogLin_opt_10000_5_4q_mnl_logdistabove1.0_A100.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'add100' > output_log/new/BLP_opt_10000_5_4q_mnl_logdistabove1.0_A100.txt & 

nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'add100' > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0_A100.txt &
nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'add100' > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove1.0_A100.txt &
nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'add100' > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove1.0_A100.txt & 

# ===========================================================================================

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove0.5" > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove0.5.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.0" > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.6.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove0.5" > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove0.5.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.0" > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove1.6.txt &

# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove0.5" > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove0.5.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.0" > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove1.0.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove1.6.txt & 


# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove0.5" 'replace100' > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove0.5_R100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'replace100' > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0_R100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.6" 'replace100' > output_log/new/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.6_R100.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove0.5" 'replace100' > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove0.5_R100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'replace100' > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove1.0_R100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.6" 'replace100' > output_log/new/LogLin_eval_10000_5_4q_mnl_logdistabove1.6_R100.txt &

# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove0.5" 'replace100' > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove0.5_R100.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'replace100' > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove1.0_R100.txt & 
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.6" 'replace100' > output_log/new/BLP_eval_10000_5_4q_mnl_logdistabove1.6_R100.txt & 

# ===========================================================================================

# nohup python3 summary.py 10000 5 4 mnl "logdistabove0.5" > output_log/new/summary_10000_5_4q_mnl_logdistabove0.5.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.0" > output_log/new/summary_10000_5_4q_mnl_logdistabove1.0.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.6" > output_log/new/summary_10000_5_4q_mnl_logdistabove1.6.txt & 

# nohup python3 summary.py 10000 5 4 mnl "logdistabove0.5" 'replace100' > output_log/new/summary_10000_5_4q_mnl_logdistabove0.5_R100.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.0" 'replace100' > output_log/new/summary_10000_5_4q_mnl_logdistabove1.0_R100.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.6" 'replace100' > output_log/new/summary_10000_5_4q_mnl_logdistabove1.6_R100.txt & 

# ===========================================================================================

