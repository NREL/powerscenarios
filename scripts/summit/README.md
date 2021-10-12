# Running ExaGO for constructing importance Distribution on Summit

*Date of Creation:* Tuesday, October 12, 2021

These instructions only apply to the `summit` branch of Powerscenarios and have
been tested using the `develop` branch of ExaGO with the commit hash
`86ce37e54575607e24668938795b3c90130d5fb1`. There is no reason why this
implementation may not work with newer commit of ExaGO; infact they do work
locally on a Mac. However, I have not  tested for it on Summit. All the files
necessary for doing our portion of the end to end run are present in this
directory, i.e., no complications due to HiOP and CUDA version compatibility.


## Installing ExaGO

At the end of FY'21, `ExaSGD_Spack` does not have the capability build ExaGO
with Python enabled. Thus, we need to build ExaGO separately after having all of
the dependencies (and also an extraneous ExaGO) installed using an
`ExaSGD_Spack` environment for Summit. Since the modules on Summit are always
changing, I have not added a spack environment YAML file. However, I used the
environment `ExaSGD_Spack/environments/summit/exago-develop-hiop-v0-4-1.yaml` to
install the the dependencies. This environment file will also install ExaGO but,
like I said earlier, we will ignore this version as this does not have the
Python wrappers enabled. Please follow the instructions in the `ExaSGD_Spack`
repository for installing and environment. When building the spack environment,
please make sure that you have a CoinHSL tarball with the following naming
convention `coinhsl-archive-2015.06.23.tar.gz` in the same current working
directory from where you call `spack install`. Otherwise, the spack installation
will fail.

Once we have the spack environment installed, run the command
```
[you@summit]$ spack load
```
This will prepend all of your available spack installed modules to your
`$MODULEPATH` whenever you source `spack/share/spack/setup-env.sh`. A warning,
`spack load` will show ALL available modules installed by ALL spack environments
that you have currently. If you find yourself in this unfortunate situation, use
the command
```
[you@summit]$ spack locate -i ${PACKAGE_NAME}
```
to find the correct module pertaning to your spack environment.

Now that we have all of the ExaGO dependencies installed, we load all of these
modules and install ExaGO using the familiar command
```
[you@summit]$ cmake .. \
  -DCMAKE_INSTALL_PREFIX=${DESIRED_PATH}/install \
  -DEXAGO_ENABLE_GPU=OFF \
  -DEXAGO_ENABLE_HIOP=OFF \
  -DEXAGO_ENABLE_CUDA=OFF \
  -DEXAGO_ENABLE_IPOPT=ON \
  -DEXAGO_ENABLE_MPI=ON \
  -DEXAGO_ENABLE_PETSC=ON \
  -DEXAGO_ENABLE_RAJA=OFF \
  -DEXAGO_TEST_WITH_BSUB=ON \
  -DEXAGO_ENABLE_PYTHON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release
[you@summit]$ make install
```
We will have to prepend this to our `${PATH}` if we want to access it neatly.

## Running Powerscenarios-ExaGO

Please look at the example batch file `generate_scenarios.lsf` in this directory
for reference. The user will need to create an file similar to
`setup-environment-offline.sh` that loads all the modules and paths on to the
compute nodes for the run. Looking at `setup-environment-offline.sh` in this
directory, we see that we first source our `.bash_profile`, this is necessary
in order to load a conda environment correctly later on the compute node. Next,
we load all of the available system modules followed by modules of all of the
installed spack packages. We then need to prepend our `$PYTHONPATH` with the
location of the ExaGO python wrapper, `$PATH` with the location of ExaGO binaries
and `PYWTK_CACHE_DIR` for WTK.

After this, its simply a case of running `generate_scenarios_mpi.py` using the
desired number of MPI ranks.
