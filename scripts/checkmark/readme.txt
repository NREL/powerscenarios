### to run on eagle 
module load conda
source activate py3n
mpirun -n 4 python checkmark_mpi.py config_checkmark_mpi_eagle.yml

### to run on mac
mpirun -n 4 python checkmark_mpi.py config_checkmark_mpi_mac.yml


