# Produce the distance between each tract and the nearest pharmacy.
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scipy.spatial import cKDTree

datadir = "/export/storage_covidvaccine/Data"


# read in tract centroids
tracts = gpd.read_file(f"{datadir}/Raw/popctr_tracts2010/popctr_tracts2010.shp")
tracts = tracts.loc[tracts['STATE'] == '06', :]
tracts.rename(columns={'FIPS': 'tract'}, inplace=True)
tracts = tracts[['tract', 'geometry']]

# Save the original CRS of tracts
original_crs = tracts.crs

# Project tracts to UTM CRS
avg_longitude = tracts.geometry.unary_union.centroid.x
utm_zone = int((avg_longitude + 180) / 6) + 1
crs = f'EPSG:326{str(utm_zone).zfill(2)}'  # This is the EPSG code for UTM zones in the northern hemisphere
tracts = tracts.to_crs(crs)

# Read in pharmacy locations
pharm = pd.read_csv(f"{datadir}/Raw/00_Pharmacies.csv")
pharm = pharm.loc[pharm['state'].isin(['CA']), ['state', 'latitude', 'longitude']] 
pharm['geometry'] = [Point(xy) for xy in zip(pharm.longitude, pharm.latitude)]
pharm = gpd.GeoDataFrame(pharm, geometry='geometry')

# Set the CRS of pharm to the original CRS of tracts
pharm.crs = original_crs
pharm = pharm.to_crs(crs)

# Function to compute nearest distance and location
def calculate_nearest(row, destination):
    # Create tree from the destination points
    destination_points = np.array([np.array(point.coords[0]) for point in destination.geometry])
    tree = cKDTree(destination_points)
    # Find the nearest point and return the corresponding value
    distance, idx = tree.query(np.array(row['geometry'].coords[0]), k=1)
    nearest_location = destination.iloc[idx]['geometry']
    return pd.Series({'dist': distance, 'pharm_loc': nearest_location})


# tracts.iloc[:10, :].apply(calculate_nearest, destination=pharm, axis=1)

# Calculate the nearest distance and location
tracts[['dist', 'pharm_loc']] = tracts.apply(calculate_nearest, destination=pharm, axis=1)

# Convert the distance to kilometers
tracts['dist'] = tracts['dist'] / 1000
print(tracts['dist'].describe())
tracts[['tract', 'dist']].to_csv(f"{datadir}/Intermediate/tract_nearest_dist.csv", index=False)
tracts.to_csv(f"{datadir}/Intermediate/tract_nearest_pharmloc.csv", index=False)

# tracts = pd.read_csv(f"{datadir}/Intermediate/tract_nearest_dist.csv")

