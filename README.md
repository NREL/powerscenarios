powerscenarios
=====================

Renewable energy scenarios for stochastic grid operation problems

## how to install

* create a conda environment and install pywtk, powerscenarios

```bash
git clone https://github.com/NREL/pywtk.git
git clone git@github.nrel.gov:isatkaus/powerscenarios.git
cd powerscenarios
conda env update -n powerscenarios
source activate powerscenarios
cp patches/fill_tz.py ../pywtk/pywtk/  # needed patch for pywtk if using py3, or change print statements manually
cd ../pywtk/
python setup.py install
cd ../powerscenarios/
pip install -e .
```


## optional packages
```bash
conda install -c conda-forge gmaps # for visualization of grids (buses, wind sites, power lines, etc)
pip install cufflinks # for quick scenario plotting 
```

## data dir

* set environment variable PYWTK_CACHE_DIR to where WIND Toolkit data is/should be stored 

* if data is missing it will be dowloaded from AWS

* e.g. on a local machine add to .bash_profile:
```bash
export PYWTK_CACHE_DIR=${HOME}/pywtk-data
```
* on Summit data is located at
```bash
${PROJWORK}/csc359/pywtk-data  
```

## notebooks/ contains examples to get started

* notebook generate_scenarios.ipynb generates scenarios for TAMU or RTS grids

## scripts/ 
* modify config.yaml as needed and run

```bash
python generate_scenrios.py config.yaml
```


