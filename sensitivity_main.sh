##### SENSITIVITY ANALYSIS #####

## DEMAND ESTIMATION
# nohup python3 Demand/demest_assm.py 8000 5 4 mnl > output_log/sensitivity/Demand_8000_5_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 12000 5 4 mnl > output_log/sensitivity/Demand_12000_5_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 10000 10 4 mnl > output_log/sensitivity/Demand_10000_10_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/Demand_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/Demand_10000_5_4q_mnl_logdistabove0.8.txt &


## MATRIX INITIALIZATION
# nohup python3 initialization.py Dollar 8000 5 4 mnl > output_log/sensitivity/init_8000_5_4q_mnl.txt &
# nohup python3 initialization.py Dollar 12000 5 4 mnl > output_log/sensitivity/init_12000_5_4q_mnl.txt &
# nohup python3 initialization.py Dollar 10000 10 4 mnl > output_log/sensitivity/init_10000_10_4q_mnl.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/init_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/init_10000_5_4q_mnl_logdistabove0.8.txt &


## OPTIMIZATION
# nohup python3 main.py MNL_partial Dollar 8000 5 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.6" > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove0.8" > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_logdistabove0.8.txt &


## EVALUATION
# nohup python3 main.py MNL_partial Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove0.8" > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_logdistabove0.8.txt &


## BASELINE EVALUATION (PHARMACY-ONLY)
# nohup python3 main.py MaxVaxDistLogLin Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/LogLin_eval_8000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/LogLin_eval_12000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/LogLin_eval_10000_10_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/LogLin_eval_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove0.8" > output_log/sensitivity/LogLin_eval_10000_5_4q_mnl_logdistabove0.8.txt &


## SUMMARY
# nohup python3 summary.py 8000 5 4 mnl > output_log/sensitivity/leftover/summary_8000_5_4q_mnl.txt & 
# nohup python3 summary.py 12000 5 4 mnl > output_log/sensitivity/leftover/summary_12000_5_4q_mnl.txt & 
# nohup python3 summary.py 10000 10 4 mnl > output_log/sensitivity/leftover/summary_10000_10_4q_mnl.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/leftover/summary_10000_5_4q_mnl_logdistabove1.6.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/leftover/summary_10000_5_4q_mnl_logdistabove0.8.txt & 