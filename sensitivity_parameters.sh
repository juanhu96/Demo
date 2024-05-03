#!/bin/bash

# capacities=(8000 12000 10000)
# Ms=(5 5 10)
# nsplitses=(4 4 4)

capacities=(10000)
Ms=(10)
nsplitses=(4)

array_length=${#capacities[@]}

for (( i=0; i<${array_length}; i++ ))
do
    capacity=${capacities[$i]}
    M=${Ms[$i]}
    nsplits=${nsplitses[$i]}

    init_cmd="nohup python3 initialization.py Dollar $capacity $M $nsplits mnl > output_log/Dollar/init_${capacity}_${M}_${nsplits}q_mnl.txt &"
    
    opt_cmd="nohup python3 main.py MNL_partial_new Dollar $capacity $M $nsplits optimize mnl > output_log/Dollar/MNL_partial_new_opt_${capacity}_${M}_${nsplits}q_mnl.txt &"
    old_opt_cmd="nohup python3 main.py MNL_partial Dollar $capacity $M $nsplits optimize mnl > output_log/Dollar/MNL_partial_opt_${capacity}_${M}_${nsplits}q_mnl.txt &"
    
    eva_cmd="nohup python3 main.py MNL_partial_new Dollar $capacity $M $nsplits evaluate mnl > output_log/Dollar/MNL_partial_new_eval_${capacity}_${M}_${nsplits}q_mnl_fcfs.txt &"
    old_eva_cmd="nohup python3 main.py MNL_partial Dollar $capacity $M $nsplits evaluate mnl > output_log/Dollar/MNL_partial_eval_${capacity}_${M}_${nsplits}q_mnl_fcfs.txt &"
    
    pharmacy_eva_cmd="nohup python3 main.py MaxVaxDistLogLin Dollar $capacity $M $nsplits evaluate mnl > output_log/Pharmacy/Pharmacy_eval_${capacity}_${M}_${nsplits}q_mnl_fcfs.txt &"
    summary_cmd="nohup python3 summary.py $capacity $M $nsplits mnl > output_log/Dollar/summary_${capacity}_${M}_${nsplits}q_mnl.txt &"

    eval $summary_cmd
done

# =====================================================================================

# logdistaboves=("0.8" "1.6")

# for logdistabove in "${logdistaboves[@]}"
# do
#     init_cmd="nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove${logdistabove}" > "output_log/Dollar/init_10000_5_4q_mnl_logdistabove${logdistabove}.txt" &"
#     opt_cmd="nohup python3 main.py MNL_partial_new Dollar 10000 5 4 optimize mnl "logdistabove${logdistabove}" > "output_log/Dollar/MNL_partial_new_opt_10000_5_4q_mnl_logdistabove${logdistabove}.txt" &"
#     eva_cmd="nohup python3 main.py MNL_partial_new Dollar 10000 5 4 evaluate mnl "logdistabove${logdistabove}" > "output_log/Dollar/MNL_partial_new_eval_10000_5_4q_mnl_logdistabove${logdistabove}.txt" &"
#     pharmacy_eva_cmd="nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove${logdistabove}" > "output_log/Pharmacy/Pharmacy_eval_10000_5_4q_mnl_logdistabove${logdistabove}_fcfs.txt" &"
#     summary_cmd="nohup python3 summary.py 10000 5 4 mnl "logdistabove${logdistabove}" > output_log/Dollar/summary_10000_5_4q_mnl_logdistabove${logdistabove}.txt &"

#     eval $summary_cmd
# done

# =====================================================================================

# python3 merge_files.py 10000 5 4 mnl