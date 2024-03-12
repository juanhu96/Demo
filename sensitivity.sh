### SENSITIVITY ANALYSIS ###

## CAPACITY
# nohup python3 Demand/demest_assm.py 8000 5 4 mnl > output_log/sensitivity/Demand_8000_5_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 12000 5 4 mnl > output_log/sensitivity/Demand_12000_5_4q_mnl.txt &

## CONSIDERATION SET
# nohup python3 Demand/demest_assm.py 10000 10 4 mnl > output_log/sensitivity/Demand_10000_10_4q_mnl.txt &

## DISTANCE FORM
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/Demand_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/Demand_10000_5_4q_mnl_logdistabove0.8.txt &

################

# nohup python3 initialization.py Dollar 8000 5 4 mnl > output_log/sensitivity/init_8000_5_4q_mnl.txt &
# nohup python3 initialization.py Dollar 12000 5 4 mnl > output_log/sensitivity/init_12000_5_4q_mnl.txt &
# nohup python3 initialization.py Dollar 10000 10 4 mnl > output_log/sensitivity/init_10000_10_4q_mnl.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/init_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/init_10000_5_4q_mnl_logdistabove0.8.txt &

################

# nohup python3 main.py MNL_partial Dollar 8000 5 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.6" > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove0.8" > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_logdistabove0.8.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add100' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add200' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A200.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add300' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A300.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add400' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A400.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add500' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A500.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add600' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A600.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add700' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A700.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add800' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A800.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add900' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A900.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add1000' > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_A1000.txt &

################

# nohup python3 main.py MNL_partial Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove0.8" > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_logdistabove0.8.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/LogLin_eval_8000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/LogLin_eval_12000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/LogLin_eval_10000_10_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/LogLin_eval_10000_5_4q_mnl_logdistabove1.6.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add100' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A100.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add200' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A200.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add300' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A300.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add400' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A400.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add500' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A500.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add600' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A600.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add700' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A700.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add800' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A800.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add900' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A900.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add1000' > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_A1000.txt &

# nohup python3 summary.py 10000 5 4 mnl 'add100' > output_log/sensitivity/summary_10000_5_4q_mnl_A100.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add200' > output_log/sensitivity/summary_10000_5_4q_mnl_A200.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add300' > output_log/sensitivity/summary_10000_5_4q_mnl_A300.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add400' > output_log/sensitivity/summary_10000_5_4q_mnl_A400.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add500' > output_log/sensitivity/summary_10000_5_4q_mnl_A500.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add600' > output_log/sensitivity/summary_10000_5_4q_mnl_A600.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add700' > output_log/sensitivity/summary_10000_5_4q_mnl_A700.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add800' > output_log/sensitivity/summary_10000_5_4q_mnl_A800.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add900' > output_log/sensitivity/summary_10000_5_4q_mnl_A900.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add1000' > output_log/sensitivity/summary_10000_5_4q_mnl_A1000.txt & 


# LEFTOVERS
# nohup python3 main.py MNL_partial Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/leftover/MNL_partial_eval_leftover_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/leftover/MNL_partial_eval_leftover_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/leftover/MNL_partial_eval_leftover_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/leftover/MNL_partial_eval_leftover_10000_5_4q_mnl_logdistabove1.6.txt &

# nohup python3 main.py MaxVaxDistLogLin Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/leftover/LogLin_eval_leftover_8000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/leftover/LogLin_eval_leftover_12000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/leftover/LogLin_eval_leftover_10000_10_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/leftover/LogLin_eval_leftover_10000_5_4q_mnl_logdistabove1.6.txt &


################

# nohup python3 summary.py 8000 5 4 mnl > output_log/sensitivity/leftover/summary_8000_5_4q_mnl.txt & 
# nohup python3 summary.py 12000 5 4 mnl > output_log/sensitivity/leftover/summary_12000_5_4q_mnl.txt & 
# nohup python3 summary.py 10000 10 4 mnl > output_log/sensitivity/leftover/summary_10000_10_4q_mnl.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/leftover/summary_10000_5_4q_mnl_logdistabove1.6.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/leftover/summary_10000_5_4q_mnl_logdistabove0.8.txt & 


################
# RANDOMIZATION

# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add100' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add200' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A200.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add300' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A300.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add400' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A400.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add500' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A500.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add600' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A600.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add700' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A700.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add800' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A800.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add900' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A900.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add1000' > output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A1000.txt &

# nohup python3 summary.py 10000 5 4 mnl 'add100' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A100.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add200' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A200.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add300' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A300.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add400' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A400.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add500' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A500.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add600' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A600.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add700' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A700.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add800' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A800.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add900' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A900.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add1000' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A1000.txt & 