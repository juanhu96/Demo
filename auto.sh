###

### M, K

### ESTIMATION
nohup python3 Demand/demest_assm.py 10000 5 4 mnl > output_log/Demand_.txt


### OPTIMIZATION
nohup python3 initialization.py Dollar 10000 5 4 mnl > output_log/Dollar/init_10000_5_4q_mnl.txt
nohup python3 main.py MNL_partial_new Dollar 10000 5 4 optimize mnl > output_log/Dollar/MNL_partial_new_opt_10000_5_4q_mnl.txt
nohup python3 main.py MNL_partial_new Dollar 10000 5 4 evaluate mnl > output_log/Dollar/MNL_partial_new_eval_10000_5_4q_mnl.txt