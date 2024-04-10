# No pain, all gain?

In this project, I analyse the Automation of Reports and Consolidated Orders System (ARCOS) dataset from the Drug Enforcement Agency (DEA). This dataset contains info on shipments of synthetic opioids from manufacturers to buyers (pharmacies/practitioners) from 2006-2019. I supplement with [geospatial](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html) and [descriptive](https://corgis-edu.github.io/corgis/python/county_demographics/) data from the US Census. There is no need to download these beforehand, as download can be done via [main.ipynb](main.ipynb).

I am specifically interested in the period around the introduction of a abuse-deterrent version of the OxyContin drug in 2010. The results of the project can be seen from running [main.ipynb](main.ipynb).

## Dependencies:

I have included all dependencies in the `env.yaml`-file. To replicate this repository, kindly do the following:
1) Clone this repository.
2) Open your `conda`/`mamba` prompt.
3) Install from the `env.yaml`-file by writing: `conda env create -f env.yaml`

## Attribution
I draw on some of the data processing done by Washington Post found [here](https://github.com/wpinvestigative/arcos-api). I thank them for making their datasets publicly available.