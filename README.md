# Closer to Home: An Estimate-then-Optimize Approach to Improve Access to Healthcare Services

## Data sources
 - Vaccination rates: [CA HHS Portal](https://data.chhs.ca.gov/dataset/covid-19-vaccine-progress-dashboard-data-by-zip-code) [Legacy table based on age 5+ denominators up to July 12, 2022](https://data.chhs.ca.gov/dataset/ead44d40-fd63-4f9f-950a-3b0111074de8/resource/580fb3a5-2451-4e18-be50-5a22f1ee7341/download/covid19vaccinesbyzipcode_071222.csv)
 - [HPI](https://api.healthyplacesindex.org/)
   - Tract-level: 2022, imputed
   - ZIP-level: 2022, imputed with 2011 data, otherwise assigned to quartile 1
 - Block data:
   - Population aged 5+: data.census.gov [link](https://data.census.gov/table?t=Resident+Population&g=040XX00US06$1000000&y=2020&tid=DECENNIALDHC2020.P1)
   - Coordinates: [link](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/California/)
 - [`AdminShapefiles`](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
   - 2010 tract boundaries (for compatibility with HPI)
   - 2020 ZIP boundaries 
   - Tract and ZIP datasets intersected in `ziptract.py`.
 - 2016 vote shares (`ca_vest_16/`): [redistrictingdatahub.org](https://redistrictingdatahub.org/dataset/vest-2016-california-precinct-and-election-results/)
   - Vote shares are not currently used in the demand model
 - Tract demographics: [ACS 2019 5-year data](https://www.census.gov/topics/research/guidance/planning-databases.2020.html), saved in `Data/Raw/Census/`.
 - ZIP demographics: ACS 2019 5-year data pulled by Cubit.
 - Tract and ZIP health insurance data: Census table B27010.
 - `Location/01_DollarStores - 10_Libraries`: locations of the various public/private partnerships, purchased from data company
 - Centroids:
   - Tract: 2010 tract-level population centroids obtained from https://www.baruch.cuny.edu/confluence/display/geoportal/US+Census+Population+Centroids
   - ZIP: `zipcodeR` package (imported in `aux_zip.R`)

Scripts:
 - `process_raw_data.R`: processes raw distance data and builds distance matrices
 - `demest_tractdemog.py`: estimates demand
 - `main.py`: runs main optimization

<img width="1512" alt="image" src="https://github.com/zhijianli9999/VaxDemandDistance/assets/60592696/bf6df058-0bb6-4033-bdc3-ef7d42d06940">
