#!/bin/bash
# Wrapper: compile main_mpi_omp.cc, submit qsub_mpi_omp.sh, wait, print output.
# Usage:  bash run_mpi_omp.sh

# ── Compile ───────────────────────────────────────────────────────────────────
echo "[compile] mpicxx main_mpi_omp.cc -o main_mpi_omp -O2 -std=c++11 -fopenmp -lm"
mpicxx main_mpi_omp.cc -o main_mpi_omp -O2 -std=c++11 -fopenmp -lm
if [ $? -ne 0 ]; then
    echo "[error] compilation failed, aborting."
    exit 1
fi
echo "[compile] done."

# ── Submit and wait ───────────────────────────────────────────────────────────
> test_omp.o
> test_omp.e

JOB_ID=$(qsub qsub_mpi_omp.sh)
if [ $? -ne 0 ]; then
    echo "[error] qsub failed: $JOB_ID"
    exit 1
fi
echo "[submit] job $JOB_ID"

while true; do
    STATUS=$(qstat "$JOB_ID" 2>/dev/null | tail -1 | awk '{print $5}')
    if [ "$STATUS" = "C" ] || [ -z "$STATUS" ]; then
        break
    fi
    echo "[wait]   status=$STATUS ..."
    sleep 2
done

echo "[done]   job $JOB_ID completed."
echo ""
echo "======== stderr (test_omp.e) ========"
cat test_omp.e
echo ""
echo "======== stdout (test_omp.o) ========"
cat test_omp.o
