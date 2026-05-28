#!/bin/sh
# =============================================================================
# qsub_mpi.sh — PBS job script for MPI IVF-SIMD ANN benchmark
#
# Compile on master node BEFORE submitting:
#   cd ~/ann
#   mpicxx main_mpi.cc -o main_mpi -O2 -std=c++11 -lm
#
# Submit:
#   qsub qsub_mpi.sh
#
# To sweep process counts, change nodes/ppn/np together:
#
#   P=1 :  nodes=1, ppn=1, -np 1    (single-process baseline)
#   P=2 :  nodes=1, ppn=2, -np 2
#   P=4 :  nodes=1, ppn=4, -np 4
#   P=8 :  nodes=1, ppn=8, -np 8    ← default below
#   P=16:  nodes=2, ppn=8, -np 16
#
# Rule from guide:  np <= nodes × ppn,  nodes <= 4,  ppn <= 8
# Without intra-process multithreading, set np = nodes × ppn exactly.
# =============================================================================
#PBS -N qsub_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=8

# ── Distribute binary and index cache to every allocated node ─────────────────
# MPI launches one process per core; every node must have the binary locally.
# /anndata/ is shared NFS so base/query data does not need copying.
NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    scp master_ubss1:/home/${USER}/ann/main_mpi ${node}:/home/${USER}/ 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files  ${node}:/home/${USER}/ 1>&2
done

# ── Run MPI job ───────────────────────────────────────────────────────────────
# -np must match the np value you intend (change together with nodes/ppn above)
/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main_mpi

# ── Sync newly written index cache files back to master ───────────────────────
scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1

# ── Clean up compute node local storage ──────────────────────────────────────
rm    /home/${USER}/main_mpi
rm -r /home/${USER}/files/
