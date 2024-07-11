random_values=("random42" "random13" "random940" "random457" "random129")
add_values=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")

# EVALUATION
for add in "${add_values[@]}"; do
  for random in "${random_values[@]}"; do

        add_num=$(echo $add | grep -o -E '[0-9]+')
        random_num=$(echo $random | grep -o -E '[0-9]+')
        output_filename="output_log/Pharmacy/Pharmacy_eval_10000_5_4q_mnl_A${add_num}_random${random_num}_fcfs.txt" # FCFS
        # nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "$add" "$random" > $output_filename &

  done
done

# SUMMARY
for add in "${add_values[@]}"; do
  for random in "${random_values[@]}"; do

        add_num=$(echo $add | grep -o -E '[0-9]+')
        random_num=$(echo $random | grep -o -E '[0-9]+')
        output_filename="output_log/Summary/randomization/summary_10000_5_4q_mnl_A${add_num}_random${random_num}_fcfs.txt" # FCFS
        # nohup python3 summary.py 10000 5 4 mnl "$add" "$random" > $output_filename &

  done
done

# python3 merge_files.py 10000 5 4 mnl