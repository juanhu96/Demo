tablepath = "/export/storage_covidvaccine/Result/Demand/coeftables/"

def combine_latex_tables(file_paths, model_names, header, table="appendix"):
    """
    Produce LaTeX table in the paper. 
    table = "appendix" or "main"
    """

    row_names = [  #the actual names in the table
        "HPI Quantile 4", "HPI Quantile 3", "HPI Quantile 2", "HPI Quantile 1", 
        "Log(Distance) * HPI Quantile 4", "Log(Distance) * HPI Quantile 3", 
        "Log(Distance) * HPI Quantile 2", "Log(Distance) * HPI Quantile 1", 
        "Race White", "Race Black", "Race Asian", "Race Hispanic", "Race Other", 
        "Health Employer", "Health Medicare", "Health Medicaid", "Health Other", 
        "College Grad", "Unemployment", "Poverty", 
        "Log(Median Household Income)", "Log(Median Home Value)", 
        "Log(Population Density)"
    ]

    if table=="main":
        row_names += ["Constant"]
    rownames_todetect = dict([(row, [row]) for row in row_names])
    for qq in range(1, 5):
        rownames_todetect[f"Log(Distance) * HPI Quantile {qq}"] += [f"logdist abovethresh X hpi quantile{qq}"]
    rename_dict = {
        "Log(Distance) * HPI Quantile 1": "Log-distance $\\times$ HPI Quartile 1",
        "Log(Distance) * HPI Quantile 2": "Log-distance $\\times$ HPI Quartile 2",
        "Log(Distance) * HPI Quantile 3": "Log-distance $\\times$ HPI Quartile 3",
        "Log(Distance) * HPI Quantile 4": "Log-distance $\\times$ HPI Quartile 4",
        "HPI Quantile 1": "HPI Quartile 1 (least healthy)",
        "HPI Quantile 2": "HPI Quartile 2",
        "HPI Quantile 3": "HPI Quartile 3",
        "HPI Quantile 4": "HPI Quartile 4 (most healthy)",
        "Poverty": "Poverty Level",
        "College Grad": "College Graduation Rate",
        "Unemployment": "Unemployment Rate",
        "Health Employer": "Health Insurance: Employer",
        "Health Medicare": "Health Insurance: Medicare",
        "Health Medicaid": "Health Insurance: Medicaid",
        "Health Other": "Health Insurance: Other"
    }
    coef_dict = {row: [] for row in row_names}
    se_dict = {row: [] for row in row_names}
    estimate_pad = 18
    
    end_of_block_rows_for_main = ["Log(Distance) * HPI Quantile 1", "HPI Quantile 1", "Race Other", "Health Other", "Poverty", "Log(Population Density)"]
    
    # Read and parse each small table
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for row_name in row_names:

                row_is_reference = True
                for j in range(len(lines)):
                    if any([name == lines[j].split('&')[0].strip() for name in rownames_todetect[row_name]]):
                        estimate = lines[j].split('&')[1].replace('\\\\', '').strip()
                        std_error = lines[j + 1].split('&')[1].replace('\\\\', '').strip()
                        coef_dict[row_name].append(f"{estimate}".ljust(estimate_pad))
                        se_dict[row_name].append(f"{std_error}".ljust(estimate_pad))
                        row_is_reference = False
                        break
                if row_is_reference:
                    coef_dict[row_name].append("Ref.".ljust(estimate_pad))
                    se_dict[row_name].append("".ljust(estimate_pad))
    
    # Construct the LaTeX table
    if table == "appendix":
        header += " & ".join(model_names)
        header += " \\\\ \n\\midrule\n"

    rowname_pad = 40
    body = ""
    for row_name in row_names:
        row_name_disp = rename_dict.get(row_name, row_name)
        if table == "appendix":
            coefs = " & ".join(coef_dict[row_name])
            body += f"{row_name_disp.ljust(rowname_pad)} & {coefs} \\\\ \n"
            ses = " & ".join(se_dict[row_name])
            body += f"{''.ljust(rowname_pad)} & {ses} \\\\ \n\\addlinespace\n"
        else: # Main text: coef and se in same row
            body += f"{row_name_disp.ljust(rowname_pad)} & " + " & ".join([f"{coef_dict[row_name][ii]} & {se_dict[row_name][ii]}" for ii in range(len(file_paths))]) + " \\\\"
            if row_name in end_of_block_rows_for_main:
                body += "\n\\addlinespace"
            body += "\n"

    
    footer = """
\\bottomrule
\\end{tabular}
"""
    if table == "appendix":
    
        footer += """
}
\\begin{flushleft}
\\footnotesize{\\textit{Note.} ($\dagger$) denotes our main specification.}
\end{flushleft}
"""
        footer += "\\end{table}"
    
    return header + body + footer






# Combine tables for appendix
file_paths = [
    f"{tablepath}coeftable_10000_5_4q_mnl.tex", 
    f"{tablepath}coeftable_10000_10_4q_mnl.tex", 
    f"{tablepath}coeftable_8000_5_4q_mnl.tex", 
    f"{tablepath}coeftable_12000_5_4q_mnl.tex", 
    f"{tablepath}coeftable_10000_5_4q_mnl_logdistabove0.5.tex", 
    f"{tablepath}coeftable_10000_5_4q_mnl_logdistabove1.0.tex", 
    f"{tablepath}coeftable_10000_300_4q.tex"]

model_names = ["$M = 5$ ($\dagger$)", "$M = 10$", "$K = 8000$", "$K = 12000$", "$\\bar{d}=0.5$", "$\\bar{d}=1$", "Closest Location"]

header = """
\\begin{table}[htb] \\centering \\setlength{\\tabcolsep}{2pt}
{\\footnotesize
\\caption{Robustness Check: Estimation Results} \\label{table:estimation_results_robustness}
\\begin{tabular}{l""" + "c" * len(file_paths) + """}
\\toprule
& \multicolumn{2}{c}{Choice set}  & \multicolumn{2}{c}{Capacity} & \multicolumn{2}{c}{Distance form} & Appointment \\\\\n \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-8} 
 Independent Variable & """

combined_table = combine_latex_tables(file_paths, model_names, header, table="appendix")


outpath = "/export/storage_covidvaccine/Result/Demand/coeftables/combined_table.tex"
print(combined_table)
# save 
with open(outpath, 'w') as f:
    f.write(combined_table)


# Main table

header_main = """
\\begin{tabular}{lcc} 
\\toprule
& \\multicolumn{2}{c}{Model Outcome:} \\\\  
& \\multicolumn{2}{c}{\\textit{Fraction Fully Vaccinated}}    \\\\
\\cmidrule(lr){2-3}
Independent variable                   & Coef.         & Std. error \\\\
\\midrule
"""
file_paths_main = [f"{tablepath}coeftable_10000_5_4q_mnl.tex"]
outpath_main = "/export/storage_covidvaccine/Result/Demand/coeftables/combined_table_main.tex"
combined_table_main = combine_latex_tables(file_paths_main, None, header_main, table="main")
with open(outpath_main, 'w') as f:
    f.write(combined_table_main)



# Other tables (in the format of the main table)

header_main = """
\\begin{tabular}{lcc} 
\\toprule
& \\multicolumn{2}{c}{Model Outcome:} \\\\  
& \\multicolumn{2}{c}{\\textit{Fraction Fully Vaccinated}}    \\\\
\\cmidrule(lr){2-3}
Independent variable                   & Coef.         & Std. error \\\\
\\midrule
"""
file_paths_main = [f"{tablepath}coeftable_10000_1000_4q.tex"]
outpath_main = "/export/storage_covidvaccine/Result/Demand/coeftables/combined_table_10000_1000_4q.tex"
combined_table_main = combine_latex_tables(file_paths_main, None, header_main, table="main")
with open(outpath_main, 'w') as f:
    f.write(combined_table_main)



# Combine tables to report
file_paths = [
    f"{tablepath}coeftable_10000_5_4q_mnl.tex", 
    f"{tablepath}coeftable_10000_10_4q_mnl.tex", 
    f"{tablepath}coeftable_10000_300_4q_mnl.tex", 
    f"{tablepath}coeftable_10000_1000_4q_mnl.tex", 
    f"{tablepath}coeftable_10000_300_4q.tex",
    f"{tablepath}coeftable_10000_1000_4q.tex"]

model_names = ["$M = 5$ ($\dagger$)", "$M = 10$", "$M = 300$", "$M = 1000$", "$M = 300$", "$M = 1000$"]

header = """
\\begin{table}[htb] \\centering \\setlength{\\tabcolsep}{2pt}
{\\footnotesize
\\begin{tabular}{l""" + "c" * len(file_paths) + """}
\\toprule
& \multicolumn{4}{c}{MNL}  & \multicolumn{2}{c}{Closest Location}  \\\\\n \\cmidrule(lr){2-5} \\cmidrule(lr){6-7}
 Independent Variable & """

combined_table = combine_latex_tables(file_paths, model_names, header, table="appendix")

outpath = "/export/storage_covidvaccine/Result/Demand/coeftables/combined_table_aux.tex"
print(combined_table)
with open(outpath, 'w') as f:
    f.write(combined_table)
