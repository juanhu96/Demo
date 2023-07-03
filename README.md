# VaxDemandDistance

## Data sources
 - [HPI](https://api.healthyplacesindex.org/): 2022, tract level
 - [`AdminShapefiles`](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
   - 2010 tract boundaries (for compatibility with HPI)
   - 2020 ZIP boundaries 
   - Tract and ZIP datasets intersected in `ziptract.py`.
 - 2016 vote shares (`ca_vest_16/`): [redistrictingdatahub.org](https://redistrictingdatahub.org/dataset/vest-2016-california-precinct-and-election-results/)
   - Vote shares are not currently used in the demand model
 - `TRACT_merged.csv`: ACS 2019 5-year data
   - Add to this repo the script that made this.
 - `Location/01_DollarStores - 10_Libraries`: locations of the various public/private partnerships, purchased from data company

Main scripts:
 - `process_raw_data.R`: processes raw distance data and builds distance matrices
 - `demest_tractdemog.py`: estimates demand
 - `main.py`: runs main optimization

<img width="1512" alt="image" src="https://github.com/zhijianli9999/VaxDemandDistance/assets/60592696/bf6df058-0bb6-4033-bdc3-ef7d42d06940">
