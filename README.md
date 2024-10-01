## ETO Easy Implementation

This repository contains all the code used an easy implmentation package based on framework purposed in the paper: 
`Bravo, F., Gandhi, A., Hu, J., & Long, E. F. Closer to Home: An Estimate-then-Optimize Approach to Improve Access to Healthcare Services.`

### Background
The ETO framework allows decision maker to understand how travel cost affects service utilization rate per facility location and selects the best set of locations to maximize the expected demand served. For more information, please refer to the paper.


#### Reference
If you use ETO in your research, we would appreciate a citation to the following paper:
<a href="https://ssrn.com/abstract=4008669" target="_blank"> Closer to Home: An Estimate-then-Optimize Approach to Improve Access to Healthcare Services</a>
Fernanda Bravo, Ashvin Gandhi, Jingyuan Hu, Elisa Long
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
cd demo
pip install -e .              # install in editable mode
bash batch/job_template.sh    # batch run
```
To run this with your own location and facility data, please replace areas.csv, locations.csv with your own file.


#### Requirements
ETO requires Python 3.5+ and Gurobi 10.0+. For instructions on how to download and install Gurobi optimizer, [click here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer). The code should be compatible with later versions of Gurobi.


### Data Included
Data included in the repository that are used for the package:
 - [`AdminShapefiles`](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
   - 2010 tract boundaries (for compatibility with HPI)
   - 2020 ZIP boundaries 
   - Tract and ZIP datasets intersected in `ziptract.py`.
 - Tract demographics: [ACS 2019 5-year data](https://www.census.gov/topics/research/guidance/planning-databases.2020.html), saved in `Data/Raw/Census/`.
 - ZIP demographics: ACS 2019 5-year data pulled by Cubit.
 - Tract and ZIP health insurance data: Census table B27010.
 - `Location/01_DollarStores - 10_Libraries`: locations of the various public/private partnerships, purchased from data company
 - Centroids:
   - Tract: 2010 tract-level population centroids obtained from https://www.baruch.cuny.edu/confluence/display/geoportal/US+Census+Population+Centroids
   - ZIP: `zipcodeR` package (imported in `aux_zip.R`)

### Main Scripts:
The framework runs the following main pythons files in order:

Data preparation
- `Demand/datawork/block/read_block.py`: read in block coordinates and population
- `Demand/datawork/ziptract.py`: read in ZIP and tract shapefiles, make crosswalk between ZIPs and tracts
- `Demand/datawork/block/block_cw.py`: make crosswalks between blocks and ZIPs and between blocks and tracts
- `Demand/datawork/block/block_dist.py`: find distance to nearest locations for each block (requires locations.csv)
- `Demand/datawork/dist_all.py`: distance between each block and all current locations
- `Demand/datawork/block/prep_block.py`: prepare and merge block data for demand estimation
- `Demand/datawork/prep_zip.py`: subsumes prep_demest.do and trace back to raw data (requires areas.csv)

Estimation
- `Demand/demest_assm.py`: demand estimation with capacity constraints

Optimization
- `utils/initialization.py `: compute distance matrix and BLP matrix and inputs requried for optimization
- `main.py`: run optimization and pick the best set of locations and evaluate the new set of locations