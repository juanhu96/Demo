# nohup python3 summary.py 10000 5 4 flex > output_log/summary_10000_1_4q_flexthresh2_3_15.txt & 
# nohup python3 summary.py 10000 5 4 mnl flex > output_log/summary_10000_1_4q_mnl_flexthresh2_3_15.txt & 
# nohup python3 summary.py 10000 5 4 mnl flex logdistabove1.0 > output_log/summary_10000_1_4q_mnl_flexthresh2_3_15_logdistabove1.0.txt & 

# NOTE: no random term for MaxVaxDistLogLin only 
nohup python3 summary.py 10000 5 4 flex norandomterm > output_log/summary_10000_1_4q_flexthresh2_3_15_norandomterm.txt & 
nohup python3 summary.py 10000 5 4 mnl flex norandomterm > output_log/summary_10000_1_4q_mnl_flexthresh2_3_15_norandomterm.txt & 
nohup python3 summary.py 10000 5 4 mnl flex logdistabove1.0 norandomterm > output_log/summary_10000_1_4q_mnl_flexthresh2_3_15_logdistabove1.0_norandomterm.txt & 
