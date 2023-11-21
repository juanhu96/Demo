# ==================================================================================================
# tercile vs. quartiles

# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 3 True > ./output_log/summary_3q.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True > ./output_log/summary_4q.txt &

# ==================================================================================================
# Dollar, Coffee, HighSchools

# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 100 > ./output_log/summary_4q_R100.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 200 > ./output_log/summary_4q_R200.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 400 > ./output_log/summary_4q_R400.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 800 > ./output_log/summary_4q_R400.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True None > ./output_log/summary_4q_RFull.txt &

# ==================================================================================================
# BLP vs. LogLin

# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 100 > ./output_log/summary_4q_R100.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 200 > ./output_log/summary_4q_R200.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 400 > ./output_log/summary_4q_R400.txt &
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True None > ./output_log/summary_4q_RFull.txt &

# ==================================================================================================
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/utils/export_locations.py Dollar 8000 5 4 True 100 > ./output_log/100.txt & 
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/utils/export_locations.py Dollar 8000 5 4 True 200 > ./output_log/200.txt & 
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/utils/export_locations.py Dollar 8000 5 4 True 400 > ./output_log/400.txt & 


# ==================================================================================================
# heuristic
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 True 100 > ./output_log/summary_4q_R100.txt &
nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 False 100 > ./output_log/summary_4q_R100.txt &
nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 False 200 > ./output_log/summary_4q_R200.txt &
nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 False 400 > ./output_log/summary_4q_R400.txt &