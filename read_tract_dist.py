# Read the distance between each tract and the nearest pharmacy, produced by process_raw_data.R
import pandas as pd
import numpy as np

datadir = "/export/storage_covidvaccine/Data"

distmatrix = pd.read_csv(f"{datadir}/CA_dist_matrix_current.csv", header=None)
print(distmatrix.shape)
# find minimum of each column
mindist = np.min(distmatrix, axis=0) / 1000
distmatrix_id = pd.read_csv(f"{datadir}/CA_tractID.csv", header=None)

mindist_df = pd.DataFrame({'tract': distmatrix_id[0], 'dist': mindist})
mindist_df.to_csv(f"{datadir}/Intermediate/tract_nearest_dist.csv", index=False)


