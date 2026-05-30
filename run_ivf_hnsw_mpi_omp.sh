#!/bin/bash
# Compile main_ivf_hnsw_mpi_omp.cc, submit, wait, print output.
# Usage: bash run_ivf_hnsw_mpi_omp.sh

perl -i -pe 's/[^\x00-\x7F]//g' qsub_ivf_hnsw_mpi_omp.sh

echo "[compile] mpicxx main_ivf_hnsw_mpi_omp.cc -o main_ivf_hnsw_mpi_omp ..."
mpicxx main_ivf_hnsw_mpi_omp.cc -o main_ivf_hnsw_mpi_omp \
       -O2 -std=c++11 -fopenmp -lpthread -lm
if [ $? -ne 0 ]; then echo "[error] compilation failed."; exit 1; fi
echo "[compile] done."

cp main_ivf_hnsw_mpi_omp ~/ann/main_ivf_hnsw_mpi_omp

> test_ivf_hnsw_mpi_omp.o
> test_ivf_hnsw_mpi_omp.e

JOB_ID=$(qsub qsub_ivf_hnsw_mpi_omp.sh)
if [ $? -ne 0 ]; then echo "[error] qsub failed: $JOB_ID"; exit 1; fi
echo "[submit] job $JOB_ID"

while true; do
    STATUS=$(qstat "$JOB_ID" 2>/dev/null | tail -1 | awk '{print $5}')
    if [ "$STATUS" = "C" ] || [ -z "$STATUS" ]; then break; fi
    echo "[wait]   status=$STATUS ..."
    sleep 2
done

echo "[done]   job $JOB_ID completed."
echo ""
echo "======== stderr (test_ivf_hnsw_mpi_omp.e) ========"
cat test_ivf_hnsw_mpi_omp.e
echo ""
echo "======== stdout (test_ivf_hnsw_mpi_omp.o) ========"
cat test_ivf_hnsw_mpi_omp.o
