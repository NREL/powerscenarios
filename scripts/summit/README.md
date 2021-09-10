# Running ExaGO for constructing importance Distribution on Summit

This instruction is relevant only for commits before September 12 on the `summit` branch

## Preparatory Steps

1. Load in the modules `gcc/9.1.0` and `python/3.8-anaconda3`
2. Create a conda environment using the module `python/3.8-anaconda3` on summit.
3. Install powerscenarios and all its dependencies within your conda environment. 
   Make sure that you specify `CC` and `CXX` when installing `mpi4py` using pip.
   Else it will try to install its own MPI. More documentation for it is available
   on the webpage of `mpi4py`


## Interactive mode Steps

1. Launch an interactive session using the `bsub` command as mentioned in the 
   OLCF user guide.
2. Source the environment file as 
```
source ${PROJWORK}/csc359/NREL/kpanda/setup-environment.sh
```
This should load all the spack modules and environemnt variables necessary for 
running the simulations.
3. Run the example notebook as
```
jsrn -n1 python ${POWERSCENARIOS_ROOT}/scripts/summit/generate_scenarios_mpi.py
```
where `${POWERSCENARIOS_ROOT}` indicates the path to your `powerscenarios` repo

## Batch job steps

1. navigate to `${POWERSCENARIOS_ROOT}/scripts/summit/`
2. Submit a the script similar to `generate_scenarios.lsf`
