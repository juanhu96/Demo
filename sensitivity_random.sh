##### SENSITIVITY ANALYSIS #####

## RANDOM STRATEGY (42, 13, 940, 457, 129)

add_values=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")
# random_values=("random42" "random13" "random940" "random457" "random129")
random_values=("random129")

# randomize
for add in "${add_values[@]}"; do
  for random in "${random_values[@]}"; do

        add_num=$(echo $add | grep -o -E '[0-9]+')
        random_num=$(echo $random | grep -o -E '[0-9]+')

        output_filename="output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A${add_num}_random${random_num}.txt"
        nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "$add" "$random" > $output_filename &

  done
done

# summary
# for add in "${add_values[@]}"; do
#   for random in "${random_values[@]}"; do

#         add_num=$(echo $add | grep -o -E '[0-9]+')
#         random_num=$(echo $random | grep -o -E '[0-9]+')        

#         output_filename="output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A${add_num}_random${random_num}.txt"
#         nohup python3 summary.py 10000 5 4 mnl "$add" "$random" > $output_filename &

#   done
# done

# NOTE: clean up the storage after obtaining results