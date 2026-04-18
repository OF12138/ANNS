#!/bin/sh
# =============================================================================
# qsub.sh — PBS job script for running the ANN benchmark on the cluster
#
# Purpose:
#   Submitted via `qsub qsub.sh`. The PBS scheduler allocates compute node(s),
#   sets $PBS_NODEFILE to a file listing their hostnames, then runs this script.
#
# Workflow:
#   1. Create ~/  on every allocated node (pssh = parallel ssh).
#   2. Copy the pre-compiled binary 'main' from the master node's ~/ann/.
#   3. Copy any pre-built index files from ~/ann/files/ to the compute nodes.
#   4. Distribute the binary to all nodes (pscp = parallel scp).
#   5. Execute main — it reads /anndata/ (shared NFS) and writes results.
#   6. Copy output files (index files, any results written to files/) back to
#      the master node's ~/ann/files/ for retrieval.
#   7. Clean up the binary and files/ from compute node local storage.
#
# Notes:
#   - stderr of setup commands is redirected to the terminal (1>&2) to keep
#     PBS stdout (test.o) clean for the program's actual output.
#   - The final scp uses 2>&1 to capture any copy errors into test.o.
#   - Build the binary on the master with: g++ -O3 -fopenmp -mavx2 main.cc -o main
#     then place it in ~/ann/ before submitting this job.
# =============================================================================
#PBS -N qsub
#PBS -e test.e   # PBS stderr → test.e
#PBS -o test.o   # PBS stdout (program output) → test.o

# Step 1: ensure home directory exists on all compute nodes
/usr/local/bin/pssh -h $PBS_NODEFILE mkdir -p /home/${USER} 1>&2

# Step 2: copy compiled binary from master to local storage of this node
scp master_ubss1:/home/${USER}/ann/main /home/${USER} 1>&2

# Step 3: copy pre-built index files (e.g. hnsw.index) to this node
scp -r master_ubss1:/home/${USER}/ann/files/ /home/${USER}/ 1>&2

# Step 4: broadcast binary to ALL allocated nodes in parallel
/usr/local/bin/pscp -h $PBS_NODEFILE /home/${USER}/main /home/${USER} 1>&2

# Step 5: run the benchmark (reads /anndata/ NFS, uses ~/files/ for indexes)
/home/${USER}/main

# Step 6: clean up binary from compute node
rm /home/${USER}/main

# Step 7: sync any newly generated index files back to master for inspection
scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1

# Step 8: remove local files/ to free disk space on compute node
rm -r /home/${USER}/files/
