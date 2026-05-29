#!/bin/sh
# =============================================================================
# exp/qsub_mpi_omp_perf2.sh — PBS job for Plan II perf profiling
#
# Change the two lines marked "USER SETS" to sweep configurations.
# Rule: NP × T = nodes × ppn
#
# Typical sweeps (fix total workers = 8):
#   P=1, T=8 : nodes=1:ppn=8,  NP=1
#   P=2, T=4 : nodes=1:ppn=8,  NP=2
#   P=4, T=2 : nodes=1:ppn=8,  NP=4
#   P=8, T=1 : nodes=1:ppn=8,  NP=8
#
# Queries needed: P × 250. For P=8 → 2000 (max for DEEP100K gt file).
# =============================================================================
#PBS -N mpi_omp_perf2
#PBS -e test_perf2.e
#PBS -o test_perf2.o
#PBS -l nodes=1:ppn=8          # USER SETS

NP=4                            # USER SETS: MPI processes (T = total_cores / NP)
NPROBE=16

TOTAL_CORES=$(cat $PBS_NODEFILE | wc -l)
NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
PPN=$(( TOTAL_CORES / NNODES ))
T=$(( TOTAL_CORES / NP ))
export OMP_NUM_THREADS=$T

NODES_LIST=$(cat $PBS_NODEFILE | sort | uniq)

echo "======== PBS resource info ========" 1>&2
echo "  nodes        = $NNODES" 1>&2
echo "  ppn          = $PPN" 1>&2
echo "  mpi procs NP = $NP" 1>&2
echo "  omp threads  = $T" 1>&2
echo "  nprobe       = $NPROBE" 1>&2
cat $PBS_NODEFILE 1>&2
echo "===================================" 1>&2

for node in $NODES_LIST; do
    scp master_ubss1:/home/${USER}/ann/ivf_mpi_omp_perf2 ${node}:/home/${USER}/ 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files           ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np $NP -machinefile $PBS_NODEFILE \
    /home/${USER}/ivf_mpi_omp_perf2 $NPROBE

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1
rm    /home/${USER}/ivf_mpi_omp_perf2
rm -r /home/${USER}/files/
