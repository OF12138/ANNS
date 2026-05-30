#!/bin/sh
#PBS -N hnsw_partition_mpi_omp
#PBS -e test_hnsw_partition_mpi_omp.e
#PBS -o test_hnsw_partition_mpi_omp.o
#PBS -l nodes=1:ppn=8

NP=8

TOTAL_CORES=$(cat $PBS_NODEFILE | wc -l)
NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
PPN=$(( TOTAL_CORES / NNODES ))
T=$(( TOTAL_CORES / NP ))
export OMP_NUM_THREADS=$T

NODES_LIST=$(cat $PBS_NODEFILE | sort | uniq)

echo "======== PBS resource info ========" 1>&2
echo "  nodes=$NNODES  ppn=$PPN  NP=$NP  T=$T" 1>&2
cat $PBS_NODEFILE 1>&2
echo "===================================" 1>&2

for node in $NODES_LIST; do
    scp master_ubss1:/home/${USER}/ann/main_hnsw_partition_mpi_omp ${node}:/home/${USER}/ 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files                     ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np $NP -machinefile $PBS_NODEFILE \
    /home/${USER}/main_hnsw_partition_mpi_omp

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1
rm    /home/${USER}/main_hnsw_partition_mpi_omp
rm -r /home/${USER}/files/
