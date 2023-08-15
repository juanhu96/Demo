#Initial Setup
#######################################################################
#Define in .bashrc:
# export VAXdir='/mnt/staff/zhli/VaxDemandDistance'

#Running Makefile
#######################################################################
#To run this makefile, enter the following command to your shell
# make43 -f ${VAXdir}/Code/Makefile > makefilelog_vax.out &
# (to run entire makefile), or
# make43 -f ${VAXdir}/Makefile [TARGET] > makefilelog_vax.out &
# (to update only targets)
# for example:
# make43 -f ${VAXdir}/Makefile ${FRdir}/Data/Intermediate/...
# To draw graph: ```make -Bnd | make2graph > /mnt/staff/zhli/mfgraph.dot```


ifeq ($(STATA_CMD),)
STATA_CMD = nohup stata-mp -b do
endif

ifeq ($(R_CMD),)
R_CMD = R CMD BATCH
endif


codedir = ${VAXdir}
projectdir = /export/storage_covidvaccine
datadir = $(projectdir)/Data
resultdir = $(projectdir)/Result


all: $(datadir)/Analysis/Demand/tract_utils.csv
# all: $(datadir)/Result/summary_table_BLP
# TODO: demand estimation output for now, change back to optimization output later

$(datadir)/CA_demand.csv \
$(datadir)/CA_tractID.csv \
$(datadir)/CA_demand_over_5.csv \
$(datadir)/HPIQuartile_TRACT.csv \
$(datadir)/tract_centroids.csv \
$(datadir)/CA_dist_matrix_current.csv \
$(datadir)/CA_dist_matrix_CarDealers.csv \
$(datadir)/CA_dist_matrix_Coffee.csv \
$(datadir)/CA_dist_matrix_ConvenienceStores.csv \
$(datadir)/CA_dist_matrix_DiscountRetailers.csv \
$(datadir)/CA_dist_matrix_Dollar.csv \
$(datadir)/CA_dist_matrix_FastFood.csv \
$(datadir)/CA_dist_matrix_GasStations.csv \
$(datadir)/CA_dist_matrix_HighSchools.csv \
$(datadir)/CA_dist_matrix_Libraries.csv \
$(datadir)/CA_dist_matrix_Mcdonald.csv \
$(datadir)/CA_dist_matrix_PostOffices.csv: \
		$(datadir)/Raw/Location/popctr_tracts2010/ \
		$(datadir)/Raw/Location/00_Pharmacies.csv \
		$(datadir)/Raw/Location/01_DollarStores.csv \
		$(datadir)/Raw/Location/02_DiscountRetailers.csv \
		$(datadir)/Raw/Location/03_FastFood.csv \
		$(datadir)/Raw/Location/04_Coffee.csv \
		$(datadir)/Raw/Location/05_ConvenienceStores.csv \
		$(datadir)/Raw/Location/06_GasStations.csv \
		$(datadir)/Raw/Location/07_CarDealers.csv \
		$(datadir)/Raw/Location/08_PostOffices.csv \
		$(datadir)/Raw/Location/09_HighSchools.csv \
		$(datadir)/Raw/Location/09_PublicSchools.xlsx \
		$(datadir)/Raw/Location/10_Libraries.csv \
		$(datadir)/Raw/Census/pdb2020trv2_us.csv \
		$(datadir)/Raw/HPItract2022.csv \
		$(codedir)/process_raw_data.R
	$(R_CMD) $(codedir)/process_raw_data.R


$(datadir)/Intermediate/TRACT_merged.csv: \
		$(datadir)/Intermediate/tract_votes.csv \
		$(datadir)/Raw/HPI/hpi_tract_2022.csv \
		$(datadir)/tract_centroids.csv \
		$(datadir)/Raw/ACS/ACSDT5Y2019.B27010.csv \
		$(datadir)/Raw/Census/pdb2020trv2_us.csv \
		$(codedir)/export_tract.R
	$(R_CMD) $(codedir)/export_tract.R


$(datadir)/Intermediate/zip_coords.csv: \
		$(codedir)/Demand/datawork/aux_zip.R
	$(R_CMD) $(codedir)/Demand/datawork/aux_zip.R


$(datadir)/Intermediate/tract_nearest_dist.csv: \
		$(datadir)/CA_dist_matrix_current.csv \
		$(datadir)/CA_tractID.csv \
		$(codedir)/Demand/datawork/read_tract_dist.py
	python3 $(codedir)/Demand/datawork/read_tract_dist.py


$(datadir)/Intermediate/tract_zip_crosswalk.csv: \
		$(datadir)/Raw/AdminShapefiles/tl_2020_us_zcta520/ \
		$(datadir)/Raw/AdminShapefiles/tl_2010_06_tract10/ \
		$(codedir)/Demand/datawork/ziptract.py
	python3 $(codedir)/Demand/datawork/ziptract.py


$(datadir)/Intermediate/zip_votes.csv \
$(datadir)/Intermediate/tract_votes.csv: \
		$(datadir)/Raw/AdminShapefiles/tl_2020_us_zcta520/ \
		$(datadir)/Raw/AdminShapefiles/tl_2010_06_tract10/ \
		$(datadir)/Raw/ca_vest_16/ \
		$(codedir)/Demand/datawork/voteshares.py
	python3 $(codedir)/Demand/datawork/voteshares.py


$(datadir)/Intermediate/zip_health.csv: \
		$(codedir)/Demand/datawork/zip_health.R
	$(R_CMD) $(codedir)/Demand/datawork/zip_health.R


$(datadir)/Intermediate/zip_demo.csv \
$(datadir)/Intermediate/hpi_zip.csv \
$(datadir)/Intermediate/vax_panel.csv \
$(datadir)/Analysis/Demand/demest_data.csv : \
		$(datadir)/Intermediate/zip_health.csv \
		$(datadir)/Intermediate/zip_votes.csv \
		$(datadir)/Raw/California_DemographicsByZip2020.xlsx \
		$(datadir)/Raw/Vaccination/covid19vaccinesbyzipcode_071222.csv \
		$(datadir)/Raw/HPI/hpi_zip_2022.csv \
		$(datadir)/Raw/HPI/hpi_zip_2011.csv \
		$(codedir)/Demand/datawork/prep_zip.py
	python3 $(codedir)/Demand/datawork/prep_zip.py


$(datadir)/Analysis/Demand/agent_data.csv: \
		$(datadir)/Intermediate/zip_coords.csv \
		$(datadir)/Intermediate/tract_nearest_dist.csv \
		$(datadir)/Analysis/Demand/demest_data.csv \
		$(datadir)/Intermediate/tract_zip_crosswalk.csv \
		$(datadir)/Intermediate/TRACT_merged.csv \
		$(codedir)/Demand/datawork/prep_tracts.py
	python3 $(codedir)/Demand/datawork/prep_tracts.py


$(datadir)/Analysis/Demand/tract_utils.csv: \
		$(datadir)/Analysis/Demand/agent_data.csv \
		$(datadir)/Analysis/Demand/demest_data.csv \
		$(codedir)/Demand/demest_tractdemog.py
	python3 $(codedir)/Demand/demest_tractdemog.py


# OPTIMIZATION:

# Jingyuan: populations/demand, quartile, and distance matrices
$(datadir)/Result/summary_table_BLP: \
		$(datadir)/CA_demand_over_5.csv \
		$(datadir)/HPIQuartile_TRACT.csv \
		$(datadir)/CA_dist_matrix_current.csv \
		$(datadir)/CA_dist_matrix_CarDealers.csv \
		$(datadir)/CA_dist_matrix_Coffee.csv \
		$(datadir)/CA_dist_matrix_ConvenienceStores.csv \
		$(datadir)/CA_dist_matrix_DiscountRetailers.csv \
		$(datadir)/CA_dist_matrix_Dollar.csv \
		$(datadir)/CA_dist_matrix_FastFood.csv \
		$(datadir)/CA_dist_matrix_GasStations.csv \
		$(datadir)/CA_dist_matrix_HighSchools.csv \
		$(datadir)/CA_dist_matrix_Libraries.csv \
		$(datadir)/CA_dist_matrix_Mcdonald.csv \
		$(datadir)/CA_dist_matrix_PostOffices.csv \
		$(datadir)/Analysis/Demand/tract_utils.csv \
		$(codedir)/utils/create_row.py \
		$(codedir)/utils/optimize_chain.py \
		$(codedir)/utils/optimize_model.py \
		$(codedir)/utils/optimize_main.py \
		$(codedir)/main.py
	python3 $(codedir)/main.py



# TODO: Add partnership summary stuff (?)
