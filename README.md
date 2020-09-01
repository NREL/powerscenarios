powerscenarios
=====================
Realistic data-driven renewable energy scenarios for stochastic grid operation problems
<img src="/docs/images/TAMU2000.png" width="380" height="300"> <img src="/docs/images/total_wind_power.png" width="430" height="300">   

## how to install

* on Summit load conda module
```bash
module load python/3.7.0-anaconda3-5.3.0
```

* clone and install pywtk, powerscenarios

```bash
git clone https://github.com/NREL/pywtk.git
git clone https://github.com/NREL/powerscenarios.git
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

* if data is missing it will be dowloaded from AWS as needed

* e.g. on a local machine add to .bash_profile:
```bash
export PYWTK_CACHE_DIR=${HOME}/pywtk-data
```
* e.g. on Summit add to .bash_profile: 
```bash
export PYWTK_CACHE_DIR=${PROJWORK}/csc359/pywtk-data  
```

## notebooks/ contains examples to get started

* notebook generate_scenarios.ipynb generates scenarios for TAMU or RTS grids

## scripts/ 
* modify config.yaml as needed and run

```bash
python generate_scenrios.py config.yml
```

## additional reading
* Reynolds, AM., I. Satkauskas, J. Mack, D. Sigler, W. Jones. “Scenario creation and power-conditioning strategies for operating power grids with two-stage stochastic economic dispatch.” In Proceedings of the 2020 IEEE Power & Energy Society General Meeting 
* D. Sigler, J. Maack, I. Satkauskas, M. Reynolds and W. Jones, ["Scalable Transmission Expansion Under Uncertainty Using Three-stage Stochastic Optimization,"](https://ieeexplore.ieee.org/document/9087776) 2020 IEEE Power & Energy Society Innovative Smart Grid Technologies Conference (ISGT), Washington, DC, USA, 2020, pp. 1-5, doi: 10.1109/ISGT45199.2020.9087776.]


