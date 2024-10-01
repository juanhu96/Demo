## ETO Easy Implementation

This repository contains all the code used for an easy package based on the framework purposed in the paper: 

`Bravo, F., Gandhi, A., Hu, J., & Long, E. F. Closer to Home: An Estimate-then-Optimize Approach to Improve Access to Healthcare Services.`


### Background
The ETO framework helps decision-makers understand how travel costs impact service utilization rates at each facility and identifies the optimal locations to maximize expected demand. For more details, please refer to the paper.


#### Reference
If you use ETO in your research, we would appreciate a citation to the following paper:
<a href="https://ssrn.com/abstract=4008669" target="_blank"> Closer to Home: An Estimate-then-Optimize Approach to Improve Access to Healthcare Services</a>

```bibtex
@article{bravo2024closer,
  title={Closer to Home: An Estimate-then-Optimize Approach to Improve Access to Healthcare Services},
  author={Bravo, Fernanda and Gandhi, Ashvin and Hu, Jingyuan and Long, Elisa F}
  year={2024},
  note={Available at SSRN: \url{https://ssrn.com/abstract=4008669}}
}
```


#### License
This code is available under the MIT License.

Copyright (C) 2024 Fernanda Bravo, Ashvin Gandhi, Jingyuan Hu, Elisa Long

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### Installation
Run the following snippet in a Unix terminal to install ETO and complete a test run.
```bash
git clone https://github.com/juanhu96/Demo
cd Demo
pip install -e .              # install in editable mode
./eto.sh 10000 5             # run example
```
To run this with your own location and facility data, please replace `areas.csv`, `locations.csv` in the folder with your own file.


#### Requirements
ETO requires Python 3.5+ and Gurobi 10.0+. For instructions on how to download and install Gurobi optimizer, [click here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer). The code should be compatible with later versions of Gurobi.


### Data Included
Data included in the repository that are used for the package:
 - [`AdminShapefiles`](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
   - `tl_2010_06_tract10`: 2010 tract boundaries (for compatibility with HPI)
   - `tl_2020_us_zcta520`: 2020 ZIP boundaries 
   - Tract and ZIP datasets intersected in `ziptract.py`.
 - `blocks`:
   - Population aged 5+: data.census.gov [link](https://data.census.gov/table?t=Resident+Population&g=040XX00US06$1000000&y=2020&tid=DECENNIALDHC2020.P1)
   - Coordinates: [link](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/California/)
 - `tracts`:
   - `tract_centroids.csv`: 2010 tract-level population centroids obtained from https://www.baruch.cuny.edu/confluence/display/geoportal/US+Census+Population+Centroids

#### Data Structure
The data should be downloaded and arranged in the following structure to enable correct access:
```
├── Data/
│   └── Raw/
│       └── AdminShapefiles
│           └── tl_2010_06_tract10
│               └── tl_2010_06_tract10.dbf
│               └── tl_2010_06_tract10.prj
│               └── tl_2010_06_tract10.shp
│               └── tl_2010_06_tract10.shp.xml
│               └── tl_2010_06_tract10.shx
│           └── tl_2020_us_zcta520
│               └── tl_2020_us_zcta520.cpg
│               └── tl_2020_us_zcta520.dbf
│               └── tl_2020_us_zcta520.prj
│               └── tl_2020_us_zcta520.shp
│               └── tl_2020_us_zcta520.shp.ea.iso.xml
│               └── tl_2020_us_zcta520.shx
│               └── zip_ca.cpg
│               └── zip_ca.dbf
│               └── zip_ca.prj
│               └── zip_ca.shp
│               └── zip_ca.shx
│       └── blocks
│           └── ca000012020.csv
│           └── ca000022020.csv
│           └── ca000032020.csv
│           └── cageo2020.csv
│       └── tracts
│           └── tract_centroids.csv
│   └── Intermediate/ # intermediate files for input
│   └── Analysis/     # output file from estimation
```


### Main Scripts:
The framework runs the following main pythons files in order:

Data preparation
- `src/Demand/datawork/block/read_block.py`: read in block coordinates and population
- `src/Demand/datawork/ziptract.py`: read in ZIP and tract shapefiles, make crosswalk between ZIPs and tracts
- `src/Demand/datawork/block/block_cw.py`: make crosswalks between blocks and ZIPs and between blocks and tracts
- `src/Demand/datawork/block/block_dist.py`: find distance to nearest locations for each block (requires locations.csv)
- `src/Demand/datawork/dist_all.py`: distance between each block and all current locations
- `src/Demand/datawork/block/prep_block.py`: prepare and merge block data for demand estimation
- `src/Demand/datawork/prep_zip.py`: subsumes prep_demest.do and trace back to raw data (requires areas.csv)

Estimation
- `src/Demand/demest_assm.py`: demand estimation with capacity constraints

Optimization
- `src/utils/initialization.py `: compute distance matrix and BLP matrix and inputs requried for optimization
- `src/utils/main.py`: run optimization and pick the best set of locations and evaluate the new set of locations