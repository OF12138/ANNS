#!/bin/sh
# =============================================================================
# qsub_mpi_omp.sh — PBS job script for Hybrid MPI × OMP IVF-SIMD (Plan II)
#
# How to configure for experiments:
#
#   Fix total workers = 8, vary P (MPI) vs T (OMP):
#     P=1, T=8 :  nodes=1:ppn=8,  NP=1
#     P=2, T=4 :  nodes=1:ppn=8,  NP=2
#     P=4, T=2 :  nodes=1:ppn=8,  NP=4
#     P=8, T=1 :  nodes=1:ppn=8,  NP=8   ← pure MPI baseline
#
#   Fix P=4, vary T:
#     T=1 :  nodes=1:ppn=4,  NP=4
#     T=2 :  nodes=1:ppn=8,  NP=4
#     T=4 :  nodes=2:ppn=8,  NP=4
#
#   Rules:  NP × T = nodes × ppn  (no over/under-subscription)
#           nodes <= 4,  ppn <= 8
#
# Changing parameters: edit ONLY the two lines marked "USER SETS"
# =============================================================================
#PBS -N qsub_mpi_omp
#PBS -e test_omp.e
#PBS -o test_omp.o
#PBS -l nodes=1:ppn=8          # USER SETS: total physical cores = nodes × ppn

NP=4                            # USER SETS: number of MPI processes
                                 # T (OMP threads per process) = total_cores / NP

# ── Derive resource counts ────────────────────────────────────────────────────
TOTAL_CORES=$(cat $PBS_NODEFILE | wc -l)
NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
PPN=$(( TOTAL_CORES / NNODES ))
T=$(( TOTAL_CORES / NP ))
export OMP_NUM_THREADS=$T

NODES_LIST=$(cat $PBS_NODEFILE | sort | uniq)

echo "======== PBS resource info ========" 1>&2
echo "  nodes        = $NNODES" 1>&2
echo "  ppn          = $PPN" 1>&2
echo "  total cores  = $TOTAL_CORES" 1>&2
echo "  mpi procs NP = $NP" 1>&2
echo "  omp threads  = $T  (OMP_NUM_THREADS=$OMP_NUM_THREADS)" 1>&2
echo "  total workers= $(( NP * T ))" 1>&2
echo "  nodefile entries:" 1>&2
cat $PBS_NODEFILE 1>&2
echo "===================================" 1>&2

# ── Distribute binary and index cache to every allocated node ─────────────────
for node in $NODES_LIST; do
    scp master_ubss1:/home/${USER}/ann/main_mpi_omp ${node}:/home/${USER}/ 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files     ${node}:/home/${USER}/ 1>&2
done

# ── Run hybrid MPI × OMP job ─────────────────────────────────────────────────
/usr/local/bin/mpiexec -np $NP -machinefile $PBS_NODEFILE /home/${USER}/main_mpi_omp

# ── Sync index cache back to master ──────────────────────────────────────────
scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1

# ── Clean up compute node local storage ──────────────────────────────────────
rm    /home/${USER}/main_mpi_omp
rm -r /home/${USER}/files/
