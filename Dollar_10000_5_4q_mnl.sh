
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1.0" > output_log/jan22/init_10000_5_4q_mnl_logdistabove1.0.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.0" > output_log/jan22/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.0" > output_log/jan22/BLP_opt_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.0" > output_log/jan22/LogLin_opt_10000_5_4q_mnl_logdistabove1.0.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.0" > output_log/jan22/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.0" > output_log/jan22/BLP_eval_10000_5_4q_mnl_logdistabove1.0.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.0" > output_log/jan22/LogLin_eval_10000_5_4q_mnl_logdistabove1.0.txt &

# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.0" > output_log/jan22/summary_10000_5_4q_mnl_logdistabove1.0.txt & 


### ADD 100 ###
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'add100' > output_log/jan22/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.0_A100.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'add100' > output_log/jan22/BLP_opt_10000_5_4q_mnl_logdistabove1.0_A100.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl "logdistabove1.0" 'add100' > output_log/jan22/LogLin_opt_10000_5_4q_mnl_logdistabove1.0_A100.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'add100' > output_log/jan22/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.0_A100.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'add100' > output_log/jan22/BLP_eval_10000_5_4q_mnl_logdistabove1.0_A100.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.0" 'add100' > output_log/jan22/LogLin_eval_10000_5_4q_mnl_logdistabove1.0_A100.txt &

# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.0" 'add100' > output_log/jan22/summary_10000_5_4q_mnl_logdistabove1.0_A100.txt & 
# python3 utils/export_locations.py 10000 5 4 mnl "logdistabove1.0" 'add100'

# ==============================================================================================================


# nohup python3 initialization.py Dollar 10000 5 4 mnl > output_log/jan22/init_10000_5_4q_mnl.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl > output_log/jan22/MNL_partial_opt_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl > output_log/jan22/BLP_opt_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl > output_log/jan22/LogLin_opt_10000_5_4q_mnl.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl > output_log/jan22/MNL_partial_eval_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl > output_log/jan22/BLP_eval_10000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl > output_log/jan22/LogLin_eval_10000_5_4q_mnl.txt &

# nohup python3 summary.py 10000 5 4 mnl > output_log/jan22/summary_10000_5_4q_mnl.txt & 


### ADD 100 ###
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add100' > output_log/jan22/MNL_partial_opt_10000_5_4q_mnl_A100.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl 'add100' > output_log/jan22/BLP_opt_10000_5_4q_mnl_A100.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl 'add100' > output_log/jan22/LogLin_opt_10000_5_4q_mnl_A100.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add100' > output_log/jan22/MNL_partial_eval_10000_5_4q_mnl_A100.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add100' > output_log/jan22/LogLin_eval_10000_5_4q_mnl_A100.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl 'add100' > output_log/jan22/BLP_eval_10000_5_4q_mnl_A100.txt & 

# nohup python3 summary.py 10000 5 4 mnl 'add100' > output_log/jan22/summary_10000_5_4q_mnl_A100.txt & 
# python3 utils/export_locations.py 10000 5 4 mnl 'add100'


### ADD 200 ###
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl 'add200' > output_log/jan22/MNL_partial_opt_10000_5_4q_mnl_A200.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 optimize mnl 'add200' > output_log/jan22/BLP_opt_10000_5_4q_mnl_A200.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 optimize mnl 'add200' > output_log/jan22/LogLin_opt_10000_5_4q_mnl_A200.txt &

# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl 'add200' > output_log/jan22/MNL_partial_eval_10000_5_4q_mnl_A200.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl 'add200' > output_log/jan22/LogLin_eval_10000_5_4q_mnl_A200.txt &
# nohup python3 main.py MaxVaxHPIDistBLP Dollar 10000 5 4 evaluate mnl 'add200' > output_log/jan22/BLP_eval_10000_5_4q_mnl_A200.txt & 

# nohup python3 summary.py 10000 5 4 mnl 'add200' > output_log/jan22/summary_10000_5_4q_mnl_A200.txt & 
# python3 utils/export_locations.py 10000 5 4 mnl 'add200'


# ==============================================================================================================








