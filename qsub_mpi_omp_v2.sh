#!/bin/sh
# qsub_mpi_omp_v2.sh — Plan II with reordering (main_mpi_omp_v2)
# Same tuning knobs as qsub_mpi_omp.sh.
#PBS -N qsub_mpi_omp_v2
#PBS -e test_v2.e
#PBS -o test_v2.o
#PBS -l nodes=1:ppn=8          # USER SETS

NP=4                            # USER SETS: MPI processes (T = total_cores / NP)

TOTAL_CORES=$(cat $PBS_NODEFILE | wc -l)
NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
PPN=$(( TOTAL_CORES / NNODES ))
T=$(( TOTAL_CORES / NP ))
export OMP_NUM_THREADS=$T

NODES_LIST=$(cat $PBS_NODEFILE | sort | uniq)

echo "======== PBS resource info ========" 1>&2
echo "  nodes=$NNODES  ppn=$PPN  NP=$NP  T=$T  (reorder=1)" 1>&2
cat $PBS_NODEFILE 1>&2
echo "===================================" 1>&2

for node in $NODES_LIST; do
    scp master_ubss1:/home/${USER}/ann/main_mpi_omp_v2 ${node}:/home/${USER}/ 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files         ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np $NP -machinefile $PBS_NODEFILE /home/${USER}/main_mpi_omp_v2

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1
rm    /home/${USER}/main_mpi_omp_v2
rm -r /home/${USER}/files/
