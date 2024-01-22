# =============================================

# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove1" > output_log/Demand_10000_5_4q_mnl_logdistabove1.0.txt
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1" > output_log/init_10000_5_4q_mnl_logdistabove1.0.txt 

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1" > output_log/LogLin_opt_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1" > output_log/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1" > output_log/LogLin_eval_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1" > output_log/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0.txt &

# nohup python3 summary.py 10000 5 4 mnl "logdistabove1" > output_log/summary_10000_5_4q_mnl_logdistabove1.0.txt & 

# =============================================
# VARIOUS R

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1" 'replace100' > output_log/LogLin_opt_10000_5_4q_mnl_logdistabove1.0_R100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1" 'replace100' > output_log/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0_R100.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1" 'replace400' > output_log/LogLin_opt_10000_5_4q_mnl_logdistabove1.0_R400.txt &
nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1" 'replace400' > output_log/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0_R400.txt &


nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1" 'replace100' > output_log/LogLin_eval_10000_5_4q_mnl_logdistabove1.0_R100.txt &
nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1" 'replace100' > output_log/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0_R100.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1" 'replace400' > output_log/LogLin_eval_10000_5_4q_mnl_logdistabove1.0_R400.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1" 'replace400' > output_log/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0_R400.txt &


# nohup python3 summary.py 10000 5 4 mnl "logdistabove1"  'replace100' > output_log/summary_10000_5_4q_mnl_logdistabove1.0_R100.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1"  'replace400' > output_log/summary_10000_5_4q_mnl_logdistabove1.0_R400.txt & 

# =============================================
# loglintemp
nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1" loglintemp > output_log/LogLin_opt_10000_5_4q_mnl_logdistabove1.0_loglintemp.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1" loglintemp > output_log/LogLin_eval_10000_5_4q_mnl_logdistabove1.0_loglintemp.txt &