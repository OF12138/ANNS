#!/bin/bash
# Compile exp/ivf_mpi_omp_perf2.cc, submit PBS job, wait, print output.
# Usage:  bash exp/run_mpi_omp_perf2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[compile] mpicxx exp/ivf_mpi_omp_perf2.cc -o ivf_mpi_omp_perf2 ..."
mpicxx exp/ivf_mpi_omp_perf2.cc -o ivf_mpi_omp_perf2 \
       -O2 -std=c++11 -fopenmp -I. -lm
if [ $? -ne 0 ]; then
    echo "[error] compilation failed, aborting."
    exit 1
fi
echo "[compile] done → ./ivf_mpi_omp_perf2"

cp ivf_mpi_omp_perf2 ~/ann/ivf_mpi_omp_perf2

perl -i -pe 's/[^\x00-\x7F]//g' exp/qsub_mpi_omp_perf2.sh

> test_perf2.o
> test_perf2.e

JOB_ID=$(qsub exp/qsub_mpi_omp_perf2.sh)
if [ $? -ne 0 ]; then
    echo "[error] qsub failed: $JOB_ID"
    exit 1
fi
echo "[submit] job $JOB_ID"

while true; do
    STATUS=$(qstat "$JOB_ID" 2>/dev/null | tail -1 | awk '{print $5}')
    if [ "$STATUS" = "C" ] || [ -z "$STATUS" ]; then break; fi
    echo "[wait]   status=$STATUS ..."
    sleep 2
done

echo "[done]   job $JOB_ID completed."
echo ""
echo "======== stderr (test_perf2.e) ========"
cat test_perf2.e
echo ""
echo "======== stdout (test_perf2.o) ========"
cat test_perf2.o
