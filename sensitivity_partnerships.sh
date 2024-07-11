# PHARMACY-ONLY
eval_cmd="nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl > output_log/Pharmacy/Pharmacy_eval_10000_5_4q_mnl_fcfs.txt &"

# PHARMACY + DOLLAR (FULL REPLACEMENT)
opt_cmd="nohup python3 main.py MNL_partial_new Dollar 10000 5 4 optimize mnl > output_log/Dollar/MNL_partial_new_opt_10000_5_4q_mnl.txt &"
eval_cmd="nohup python3 main.py MNL_partial_new Dollar 10000 5 4 evaluate mnl > output_log/Dollar/MNL_partial_new_eval_10000_5_4q_mnl_fcfs.txt &"

summary_cmd="nohup python3 summary.py 10000 5 4 main mnl > output_log/Summary/summary_10000_5_4q_mnl.txt &"

# eval $summary_cmd

# =====================================================================================
# OPTIMALLY ADD A STORES

declare -a types=("Dollar" "Coffee" "HighSchools")
declare -a adds=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")

for type in "${types[@]}"; do

    init_cmd="nohup python3 initialization.py $type 10000 5 4 mnl > output_log/$type/init_10000_5_4q_mnl.txt &"
    # eval $init_cmd

    for add in "${adds[@]}"; do

        add_num=$(echo $add | grep -o -E '[0-9]+')
        opt_cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 optimize mnl '$add' > output_log/$type/MNL_partial_new_opt_10000_5_4q_mnl_A${add_num}.txt &"
        eval_cmd="nohup python3 main.py MNL_partial_new $type 10000 5 4 evaluate mnl '$add' > output_log/$type/MNL_partial_new_eval_10000_5_4q_mnl_A${add_num}_fcfs.txt &"
        # eval $eval_cmd

    done
done


# SUMMARY
for add in "${adds[@]}"; do
    add_num=$(echo $add | grep -o -E '[0-9]+')
    output_filename="output_log/Summary/partnerships/summary_10000_5_4q_mnl_A${add_num}.txt"
    # nohup python3 summary.py 10000 5 4 partnerships mnl "$add" > $output_filename &
done

# =====================================================================================

# python3 merge_files.py 10000 5 4 partnerships mnl

python3 utils/export_locations.py Dollar 10000 5 4 mnl add500
python3 utils/export_locations.py Coffee 10000 5 4 mnl add500
python3 utils/export_locations.py HighSchools 10000 5 4 mnl add500