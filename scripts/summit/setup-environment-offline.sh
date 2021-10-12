#!/usr/bin/bash

source /autofs/nccs-svm1_home1/kpanda/.bash_profile

# Set the gcc compilers
module load gcc/9.1.0
module load cmake/3.20.2
module load openblas/0.3.15-omp
export CC=gcc
export CXX=g++

# Setup the python environment
module load python/3.8-anaconda3
conda activate pyego

# Load everything else that may come in ahnd for subsequent builds
source /gpfs/alpine/csc359/proj-shared/NREL/kpanda/ExaSGD_Spack/spack/share/spack/setup-env.sh
module load autoconf-archive-2019.01.06-gcc-9.1.0-ofpeiyd
module load help2man-1.47.11-gcc-9.1.0-r2fyksr
module load m4-1.4.18-gcc-9.1.0-zzafgqo
module load suite-sparse-5.8.1-gcc-9.1.0-x6oic44
module load berkeley-db-18.1.40-gcc-9.1.0-bovi5yx
module load hiop-0.4.1-gcc-9.1.0-dkfnmwt
module load magma-2.6.1-gcc-9.1.0-u2sfl76
module load superlu-dist-6.4.0-gcc-9.1.0-2zuuitt
module load bzip2-1.0.8-gcc-9.1.0-ezajxvd
module load hypre-2.20.0-gcc-9.1.0-xi7llju
module load mpfr-4.0.2-gcc-9.1.0-kvlx5zf
module load tar-1.32-gcc-9.1.0-ou5bqhh
module load coinhsl-2019.05.21-gcc-9.1.0-7oylnqm
module load ipopt-3.12.10-gcc-9.1.0-daeibuu
module load ncurses-6.2-gcc-9.1.0-4s5tbmu
module load umpire-4.1.2-gcc-9.1.0-vighw5l
module load libiconv-1.16-gcc-9.1.0-ckwgc5m
module load parmetis-4.0.3-gcc-9.1.0-rfoptwi
module load xz-5.2.5-gcc-9.1.0-4px2pfp
module load gdbm-1.18.1-gcc-9.1.0-jl2mtpe
module load libsigsegv-2.12-gcc-9.1.0-n7hugrq
module load perl-5.32.0-gcc-9.1.0-ovuahhq
module load gettext-0.21-gcc-9.1.0-ksjoo6h
module load libtool-2.4.2-gcc-9.1.0-do6knor
module load petsc-3.14.1-gcc-9.1.0-yzrtvii
module load gmp-6.1.2-gcc-9.1.0-l4233t5
module load libxml2-2.9.10-gcc-9.1.0-5gyeqg3
module load raja-0.12.1-gcc-9.1.0-uj3xkr3

# Finally prepend path for python exago
export PYTHONPATH="/gpfs/alpine/csc359/proj-shared/NREL/kpanda/ExaGO/install/lib/python3.9/site-packages:$PYTHONPATH"
export PATH="/gpfs/alpine/csc359/proj-shared/NREL/kpanda/ExaGO/install/bin:$PATH"
export PYWTK_CACHE_DIR=/gpfs/alpine/csc359/proj-shared/pywtk-data
