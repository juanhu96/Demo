import delim /export/storage_covidvaccine/Result/Demand/violation_count_10000_5_4q_mnl_iter11.csv, clear

summ v1, d

summ v1 if v1>0, d

hist v1
hist v1 if v1>0, freq xtitle("N. assignments above capacity") ytitle("N. sites")

count if v1>0
