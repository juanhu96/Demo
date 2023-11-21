import pandas as pd 
import numpy as np 
import geopandas as gpd
from shapely.geometry import Point
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import contextily as cx

import pyblp

datadir = "/export/storage_covidvaccine/Data"
outdir = "/export/storage_covidvaccine/Result/Demand"

setting_tag = "10000_200_3q"
results = pyblp.read_pickle(f"{outdir}/results_{setting_tag}.pkl")

zip_xi = pd.DataFrame({'zip' : results.problem.unique_market_ids.flatten(), 'xi' : results.xi.flatten()})

deltas = results.compute_delta()
deltas_mean = deltas.mean(axis=0)[0]


# read in full national zips
subset_to_ca = True
if subset_to_ca:
    zips_full = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")

    zips = zips_full[['ZCTA5CE20', 'geometry']]
    zips = zips.rename(columns={'ZCTA5CE20': 'zip'})
    zips['zip'] = zips['zip'].astype(int)
    zips = zips.loc[zips['zip'].isin(zip_xi['zip'])]
    # save as geopandas dataframe
    zips.to_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/zip_ca.shp")
else:
    zips = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/zip_ca.shp")


merged = zips.merge(zip_xi, on='zip')


fig, ax = plt.subplots(figsize=(12, 12))
merged.plot(column='xi', cmap='coolwarm', linewidth=0.8, edgecolor='0.8', alpha = 0.9, legend=True, ax=ax, legend_kwds={'shrink': 0.5, 'aspect': 20})
ax.set_xlim([-118.8, -116.5])
ax.set_ylim([33.5, 35.5])
cx.add_basemap(ax, crs=merged.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

plt.savefig("xi_map.png")
