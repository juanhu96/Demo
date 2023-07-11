# Intersect ZIP and tract shapefiles. ZIPs are 2020, tracts are 2010 (for compatibility with HPI and ACS data).
# Produce a crosswalk between ZIPs and tracts.
import pandas as pd
import geopandas as gpd

datadir = "/export/storage_covidvaccine/Data"

# read in full national zips
zips = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")
zips.head()
zips = zips[['ZCTA5CE20', 'geometry']]
zips = zips.rename(columns={'ZCTA5CE20': 'zip'})
zips['zip'] = zips['zip'].astype(str)

# read in tract shapefile (CA, 2010)
tracts = gpd.read_file(f"{datadir}/Raw/AdminShapefiles/tl_2010_06_tract10/tl_2010_06_tract10.shp")
tracts = tracts[['GEOID10', 'geometry']]
tracts = tracts.rename(columns={'GEOID10': 'tract'})
tracts['tract'] = tracts['tract'].astype(str)

# intersect
tractzip = gpd.overlay(tracts, zips, how='intersection')


# compute the tract population in the particular ZIP, based on the proportion of the tract area that is in the ZIP
tractzip['area'] = tractzip.geometry.area
for col in ['tract', 'zip']:
    col_area = tractzip.groupby(col)['area'].sum()
    tractzip[f"frac_of_{col}_area"] = tractzip['area'] / tractzip[col].map(col_area)


tractzip_out = tractzip.drop(columns=['geometry']).sort_values(by=['zip', 'tract'])

tractzip_out.to_csv(f"{datadir}/Intermediate/tract_zip_crosswalk.csv", index=False)
