### to run on eagle interactively
srun --time=30 --account=exasgd --ntasks=100 --pty $SHELL
module load conda
source activate py3n
mpirun -n 100 python checkmark_mpi.py config_checkmark_mpi_eagle.yml

### to run on mac
mpirun -n 4 python checkmark_mpi.py config_checkmark_mpi_mac.yml


