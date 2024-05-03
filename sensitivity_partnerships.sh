declare -a types=("Coffee" "HighSchools")
# declare -a types=("Coffee")
declare -a adds=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")

# =====================================================================================

# FULL REPLACEMENT
# for type in "${types[@]}"; do
#     opt_cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 optimize mnl > output_log/$type/MNL_partial_new_opt_10000_5_4q_mnl.txt &"
#     eval_cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 evaluate mnl > output_log/$type/MNL_partial_new_eval_10000_5_4q_mnl_fcfs.txt &"
#     eval $eval_cmd
# done

# SUMMARY
# nohup python3 summary.py 10000 5 4 mnl > output_log/Summary/partnerships/summary_10000_5_4q_mnl.txt & 

# =====================================================================================

# DIFFERENT A
# for type in "${types[@]}"; do
#     for add in "${adds[@]}"; do
        
#         add_num=$(echo $add | grep -o -E '[0-9]+')
#         opt_cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 optimize mnl '$add' > output_log/$type/MNL_partial_new_opt_10000_5_4q_mnl_A${add_num}.txt &"
#         eval_cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 evaluate mnl '$add' > output_log/$type/MNL_partial_new_eval_10000_5_4q_mnl_A${add_num}_fcfs.txt &"
#         eval $eval_cmd

#     done
# done

# SUMMARY
# for add in "${adds[@]}"; do
#     add_num=$(echo $add | grep -o -E '[0-9]+')
#     output_filename="output_log/Summary/partnerships/summary_10000_5_4q_mnl_A${add_num}.txt"
#     nohup python3 summary.py 10000 5 4 mnl "$add" > $output_filename &
# done

# python3 merge_files.py 10000 5 4 mnl