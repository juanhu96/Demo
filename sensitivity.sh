##### SENSITIVITY ANALYSIS #####

## DEMAND ESTIMATION
# nohup python3 Demand/demest_assm.py 8000 5 4 mnl > output_log/sensitivity/Demand_8000_5_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 12000 5 4 mnl > output_log/sensitivity/Demand_12000_5_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 10000 10 4 mnl > output_log/sensitivity/Demand_10000_10_4q_mnl.txt &
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/Demand_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 Demand/demest_assm.py 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/Demand_10000_5_4q_mnl_logdistabove0.8.txt &


## MATRIX INITIALIZATION
# nohup python3 initialization.py Dollar 8000 5 4 mnl > output_log/sensitivity/init_8000_5_4q_mnl.txt &
# nohup python3 initialization.py Dollar 12000 5 4 mnl > output_log/sensitivity/init_12000_5_4q_mnl.txt &
# nohup python3 initialization.py Dollar 10000 10 4 mnl > output_log/sensitivity/init_10000_10_4q_mnl.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/init_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 initialization.py Dollar 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/init_10000_5_4q_mnl_logdistabove0.8.txt &


## OPTIMIZATION
# nohup python3 main.py MNL_partial Dollar 8000 5 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 optimize mnl > output_log/sensitivity/MNL_partial_opt_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove1.6" > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 optimize mnl "logdistabove0.8" > output_log/sensitivity/MNL_partial_opt_10000_5_4q_mnl_logdistabove0.8.txt &


## EVALUATION
# nohup python3 main.py MNL_partial Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_8000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_12000_5_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/MNL_partial_eval_10000_10_4q_mnl.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MNL_partial Dollar 10000 5 4 evaluate mnl "logdistabove0.8" > output_log/sensitivity/MNL_partial_eval_10000_5_4q_mnl_logdistabove0.8.txt &


## BASELINE EVALUATION (PHARMACY-ONLY)
# nohup python3 main.py MaxVaxDistLogLin Dollar 8000 5 4 evaluate mnl > output_log/sensitivity/LogLin_eval_8000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 12000 5 4 evaluate mnl > output_log/sensitivity/LogLin_eval_12000_5_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 10 4 evaluate mnl > output_log/sensitivity/LogLin_eval_10000_10_4q_mnl.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove1.6" > output_log/sensitivity/LogLin_eval_10000_5_4q_mnl_logdistabove1.6.txt &
# nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "logdistabove0.8" > output_log/sensitivity/LogLin_eval_10000_5_4q_mnl_logdistabove0.8.txt &


## SUMMARY
# nohup python3 summary.py 8000 5 4 mnl > output_log/sensitivity/leftover/summary_8000_5_4q_mnl.txt & 
# nohup python3 summary.py 12000 5 4 mnl > output_log/sensitivity/leftover/summary_12000_5_4q_mnl.txt & 
# nohup python3 summary.py 10000 10 4 mnl > output_log/sensitivity/leftover/summary_10000_10_4q_mnl.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove1.6" > output_log/sensitivity/leftover/summary_10000_5_4q_mnl_logdistabove1.6.txt & 
# nohup python3 summary.py 10000 5 4 mnl "logdistabove0.8" > output_log/sensitivity/leftover/summary_10000_5_4q_mnl_logdistabove0.8.txt & 


# ================================================================================================
## RANDOM STRATEGY (42, 13, 940, 457, 129)

add_values=("add100" "add200" "add300" "add400" "add500" "add600" "add700" "add800" "add900" "add1000")
random_values=("random13" "random940" "random457" "random129")

for add in "${add_values[@]}"; do
  for random in "${random_values[@]}"; do

        add_num=$(echo $add | grep -o -E '[0-9]+')
        random_num=$(echo $random | grep -o -E '[0-9]+')

        output_filename="output_log/sensitivity/randomization/LogLin_eval_10000_5_4q_mnl_A${add_num}_random${random_num}.txt"
        nohup python3 main.py MaxVaxDistLogLin Dollar 10000 5 4 evaluate mnl "$add" "$random" > $output_filename &

  done
done


# ================================================================================================
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

# ================================================================================================

## SUMMARY RESULTS

# nohup python3 summary.py 10000 5 4 mnl 'add100' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A100.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add200' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A200.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add300' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A300.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add400' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A400.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add500' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A500.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add600' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A600.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add700' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A700.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add800' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A800.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add900' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A900.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add1000' > output_log/sensitivity/randomization/summary_10000_5_4q_mnl_A1000.txt & 


# nohup python3 summary.py 10000 5 4 mnl 'add100' > output_log/sensitivity/summary_10000_5_4q_mnl_A100.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add200' > output_log/sensitivity/summary_10000_5_4q_mnl_A200.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add300' > output_log/sensitivity/summary_10000_5_4q_mnl_A300.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add400' > output_log/sensitivity/summary_10000_5_4q_mnl_A400.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add500' > output_log/sensitivity/summary_10000_5_4q_mnl_A500.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add600' > output_log/sensitivity/summary_10000_5_4q_mnl_A600.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add700' > output_log/sensitivity/summary_10000_5_4q_mnl_A700.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add800' > output_log/sensitivity/summary_10000_5_4q_mnl_A800.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add900' > output_log/sensitivity/summary_10000_5_4q_mnl_A900.txt & 
# nohup python3 summary.py 10000 5 4 mnl 'add1000' > output_log/sensitivity/summary_10000_5_4q_mnl_A1000.txt & 