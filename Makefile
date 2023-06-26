#Initial Setup
#######################################################################
#Define in .bashrc:
# export VAXdir = "/export/storage_covidvaccine"

#Running Makefile
#######################################################################
#To run this makefile, enter the following command to your shell
# make43 -f /${VAXdir}/Code/Makefile > makefilelog.out &
# (to run entire makefile), or
# make43 -f /${VAXdir}/Makefile [TARGET] > makefilelog.out &
# (to update only targets)
# for example:
# make43 -f ${VAXdir}/Makefile ${FRdir}/Data/Intermediate/...


ifeq ($(STATA_CMD),)
STATA_CMD = nohup stata-mp -b do
endif

ifeq ($(R_CMD),)
R_CMD = R CMD BATCH
endif


VAXdir = /export/storage_covidvaccine
datadir = ${VAXdir}/Data
codedir = ${VAXdir}/Code
resultdir = ${VAXdir}/Result


# TODO: We only have the demand part now. Need to add the optimization part.

all: ${resultdir}/Demand/coeftable.tex


$(datadir)/CA_demand.csv \
$(datadir)/CA_tractID.csv \
$(datadir)/CA_demand_over_5.csv \
$(datadir)/HPIQuartile_TRACT.csv \
$(datadir)/CA_dist_matrix_current.csv \
$(datadir)/CA_dist_matrix_Dollar.csv \
$(datadir)/CA_dist_matrix_Mcdonald.csv &: \
		$(datadir)/Raw/Location/ \
		$(datadir)/Raw/Census/pdb2020trv2_us.csv \
		$(codedir)/process_raw_data.R
	$(R_CMD) $(codedir)/process_raw_data.R


$(datadir)/Intermediate/tract_nearest_dist.csv: \
		$(datadir)/CA_dist_matrix_current.csv \
		$(datadir)/CA_tractID.csv \
		$(codedir)/Demand/ziptract.py
	python3 $(codedir)/Demand/ziptract.py


$(datadir)/Intermediate/tract_zip_crosswalk.csv: \
		$(datadir)/Raw/AdminShapefiles/tl_2020_us_zcta520/ \
		$(datadir)/Raw/AdminShapefiles/tl_2010_06_tract10/ \
		$(codedir)/Demand/ziptract.py
	python3 $(codedir)/Demand/ziptract.py


$(datadir)/Intermediate/zip_votes.csv: \
		$(datadir)/Raw/AdminShapefiles/tl_2020_us_zcta520/ \
		$(datadir)/Raw/AdminShapefiles/tl_2010_06_tract10/ \
		$(datadir)/Raw/ca_vest_16/ \
		$(codedir)/Demand/voteshares.py
	python3 $(codedir)/Demand/voteshares.py


$(datadir)/Analysis/Demand/MAR01_vars.dta \
$(datadir)/Analysis/Demand/demest_data.csv &: \
		$(datadir)/Intermediate/zip_votes.csv \
		$(datadir)/datadir/Raw/notreallyraw/MAR01.csv \
		$(codedir)/Demand/prep_demest.do
	$(STATA_CMD) $(codedir)/Demand/prep_demest.do


$(datadir)/Analysis/Demand/agent_data.csv: \
		$(datadir)/Intermediate/tract_nearest_dist.csv \
		$(datadir)/Analysis/Demand/demest_data.csv \
		$(datadir)/Intermediate/tract_votes.csv \
		$(datadir)/Intermediate/tract_zip_crosswalk.csv \
		$(datadir)/Raw/notreallyraw/TRACT_merged.csv \
		$(datadir)/Raw/hpi2score.csv \
		$(codedir)/Demand/prep_tracts.py
	python3 $(codedir)/Demand/prep_tracts.py


$(datadir)/Analysis/m1coefs.csv \
$(datadir)/Analysis/m2coefs.csv \
$(datadir)/Analysis/Demand/demest_results_000.pkl \
$(datadir)/Analysis/Demand/demest_results_100.pkl \
$(datadir)/Analysis/Demand/demest_results_110.pkl \
$(datadir)/Analysis/Demand/demest_results_111.pkl &: \
		$(datadir)/Analysis/Demand/agent_data.csv \
		$(datadir)/Analysis/Demand/demest_data.csv \
		$(codedir)/Demand/demest_tracts.py
	python3 $(codedir)/Demand/demest_tracts.py


$(datadir)/Analysis/Demand/marg_000.dta \
$(datadir)/Analysis/Demand/marg_100.dta \
$(datadir)/Analysis/Demand/marg_110.dta \
$(datadir)/Analysis/Demand/marg_111.dta &: \
		$(datadir)/Analysis/Demand/demest_results_000.pkl \
		$(datadir)/Analysis/Demand/demest_results_100.pkl \
		$(datadir)/Analysis/Demand/demest_results_110.pkl \
		$(datadir)/Analysis/Demand/demest_results_111.pkl \
		$(codedir)/Demand/demest_tracts_margins.py
	python3 $(codedir)/Demand/demest_tracts_margins.py


$(resultdir)/Demand/margins/pooled/hpiq0_ctrl0.png \
$(resultdir)/Demand/margins/pooled/hpiq1_ctrl0.png \
$(resultdir)/Demand/margins/byqrtl/hpiq1_ctrl0.png \
$(resultdir)/Demand/margins/byqrtl/hpiq1_ctrl1.png &: \
		$(datadir)/Analysis/Demand/marg_000.dta \
		$(datadir)/Analysis/Demand/marg_100.dta \
		$(datadir)/Analysis/Demand/marg_110.dta \
		$(datadir)/Analysis/Demand/marg_111.dta \
		$(codedir)/Demand/margins.do
	$(STATA_CMD) $(codedir)/Demand/margins.do


$(resultdir)/Demand/coeftable.tex: \
		$(datadir)/Analysis/Demand/demest_results_000.pkl \
		$(datadir)/Analysis/Demand/demest_results_100.pkl \
		$(datadir)/Analysis/Demand/demest_results_110.pkl \
		$(datadir)/Analysis/Demand/demest_results_111.pkl \
		$(codedir)/Demand/demest_tracts_table.py
	python3 $(codedir)/Demand/demest_tracts_table.py


	
