##### SENSITIVITY ANALYSIS #####

## FIXED BUDGET BY PARTNERSHIPS (TODO, NOT DONE YET)

# declare -a types=("Dollar" "Coffee" "HighSchools")
declare -a types=("Coffee" "HighSchools")
declare -a adds=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")

# optimize
for type in "${types[@]}"; do
    for add in "${adds[@]}"; do
        
        add_num=$(echo $add | grep -o -E '[0-9]+')
        cmd="nohup python3 main.py MNL_partial $type 10000 5 4 optimize mnl '$add' > output_log/$type/MNL_partial_opt_10000_5_4q_mnl_A${add_num}.txt &"
        eval $cmd

    done
done

# evaluate
for type in "${types[@]}"; do
    for add in "${adds[@]}"; do
        
        add_num=$(echo $add | grep -o -E '[0-9]+')
        cmd="nohup python3 main.py MNL_partial $type 10000 5 4 evaluate mnl '$add' > output_log/$type/MNL_partial_eval_10000_5_4q_mnl_A${add_num}.txt &"
        eval $cmd

    done
done

# summary
for add in "${add_values[@]}"; do
    add_num=$(echo $add | grep -o -E '[0-9]+')
    output_filename="output_log/sensitivity/partnerships_summary_10000_5_4q_mnl_A${add_num}.txt"
    nohup python3 summary.py 10000 5 4 mnl "$add" > $output_filename &
done