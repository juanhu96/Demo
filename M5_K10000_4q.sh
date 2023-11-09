# nohup python3 /mnt/phd/jihu/VaxDemandDistance/initialization.py Dollar 10000 4 False > ./output_log/10000_4_init.txt
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxHPIDistBLP Dollar 10000 5 4 False None > ./output_log/10000_4_opteval_blp.txt & 
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxDistLogLin Dollar 10000 5 4 False None > ./output_log/10000_4_opteval_loglinear.txt
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxHPIDistBLP Dollar 10000 5 4 False None > ./output_log/10000_4_opteval_blp_heuristic.txt 
# nohup python3 /mnt/phd/jihu/VaxDemandDistance/summary.py 4 False None > ./output_log/10000_4_summary.txt


nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxHPIDistBLP Dollar 10000 5 4 False 100 > ./output_log/10000_4_100_blp.txt & 
nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxHPIDistBLP Dollar 10000 5 4 False 200 > ./output_log/10000_4_200_blp.txt & 
nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxHPIDistBLP Dollar 10000 5 4 False 400 > ./output_log/10000_4_400_blp.txt & 
nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxDistLogLin Dollar 10000 5 4 False 100 > ./output_log/10000_4_100_loglinear.txt & 
nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxDistLogLin Dollar 10000 5 4 False 200 > ./output_log/10000_4_200_loglinear.txt & 
nohup python3 /mnt/phd/jihu/VaxDemandDistance/main.py MaxVaxDistLogLin Dollar 10000 5 4 False 400 > ./output_log/10000_4_400_loglinear.txt & 