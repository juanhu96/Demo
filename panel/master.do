// Runs the panel data analysis with the mass vaccination site rollout
global codedir /export/storage_covidvaccine/Code/VaxDemandDistance/panel
global datadir "/export/storage_covidvaccine/Data"
global outdir "/export/storage_covidvaccine/Result/Demand"
global logdir "/export/storage_covidvaccine/Logs"

do $codedir/clean_mass.do
do $codedir/prep_evstudy.do
do $codedir/evstudy.do
do $codedir/prep_match.do
do $codedir/matched_regs.do

