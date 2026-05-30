#!/bin/bash
# Compile main_mpi_omp_v2.cc, submit, wait, print output.
# Usage:  bash run_mpi_omp_v2.sh

echo "[compile] mpicxx main_mpi_omp_v2.cc -o main_mpi_omp_v2 ..."
mpicxx main_mpi_omp_v2.cc -o main_mpi_omp_v2 -O2 -std=c++11 -fopenmp -lm
if [ $? -ne 0 ]; then echo "[error] compilation failed."; exit 1; fi
echo "[compile] done."

perl -i -pe 's/[^\x00-\x7F]//g' qsub_mpi_omp_v2.sh

> test_v2.o
> test_v2.e

JOB_ID=$(qsub qsub_mpi_omp_v2.sh)
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
echo "======== stderr (test_v2.e) ========"
cat test_v2.e
echo ""
echo "======== stdout (test_v2.o) ========"
cat test_v2.o
