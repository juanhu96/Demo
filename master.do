global codedir /mnt/staff/zhli/VaxDemandDistance
do $codedir/clean_mass.do
do $codedir/prep_evstudy.do
do $codedir/evstudy.do
do $codedir/prep_match.do
do $codedir/matched_regs.do
