// Runs the panel data analysis with the mass vaccination site rollout
global codedir /export/storage_covidvaccine/Code/VaxDemandDistance/panel
do $codedir/clean_mass.do
do $codedir/prep_evstudy.do
do $codedir/evstudy.do
do $codedir/prep_match.do
do $codedir/matched_regs.do

