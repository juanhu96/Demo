# main
# nohup python3 main.py MNL_partial_new Dollar 10000 5 4 optimize mnl > output_log/Dollar/MNL_partial_new_opt_10000_5_4q_mnl.txt
# nohup python3 main.py MNL_partial_new Dollar 10000 5 4 evaluate mnl > output_log/Dollar/MNL_partial_new_eval_10000_5_4q_mnl.txt
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl > output_log/Pharmacy/MNL_partial_new_eval_10000_5_4q_mnl.txt 

# random FCFS
# nohup python3 main.py MNL_partial_new Dollar 10000 5 4 evaluate mnl > output_log/Dollar/MNL_partial_new_eval_10000_5_4q_mnl_randomFCFS.txt & 
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl > output_log/Pharmacy/MNL_partial_new_eval_10000_5_4q_mnl_randomFCFS.txt & 


# optimize
# declare -a types=("Dollar")
# declare -a adds=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")
# for type in "${types[@]}"; do
#     for add in "${adds[@]}"; do
        
#         add_num=$(echo $add | grep -o -E '[0-9]+')
#         cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 optimize mnl '$add' > output_log/$type/MNL_partial_new_opt_10000_5_4q_mnl_A${add_num}.txt &"
#         eval $cmd

#     done
# done


# evaluate
# declare -a types=("Dollar")
declare -a adds=("add100" "add200" "add300")
# for type in "${types[@]}"; do
#     for add in "${adds[@]}"; do
        
#         add_num=$(echo $add | grep -o -E '[0-9]+')
#         cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 evaluate mnl '$add' > output_log/$type/MNL_partial_new_eval_10000_5_4q_mnl_A${add_num}_fcfs.txt &" # FCFS
#         eval $cmd

#     done
# done


for add in "${adds[@]}"; do
        
    add_num=$(echo $add | grep -o -E '[0-9]+')
    # cmd="nohup python3 summary.py 10000 5 4 mnl '$add' > output_log/Summary/partnerships/summary_10000_5_4q_mnl_A${add_num}.txt &" # MIP
    cmd="nohup python3 summary.py 10000 5 4 mnl '$add' > output_log/Summary/partnerships/summary_10000_5_4q_mnl_A${add_num}_fcfs.txt &" # FCFS
    eval $cmd

done


# summary
# nohup python3 summary.py 10000 5 4 mnl > output_log/Summary/summary_10000_5_4q_mnl_new.txt & 
# python3 merge_files.py 10000 5 4 mnl