#!/bin/bash
# Compile exp/ivf_mpi_omp_perf.cc, submit PBS job, wait, print output.
# Usage:  bash exp/run_mpi_omp_perf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Compile ───────────────────────────────────────────────────────────────────
echo "[compile] mpicxx exp/ivf_mpi_omp_perf.cc -o ivf_mpi_omp_perf ..."
mpicxx exp/ivf_mpi_omp_perf.cc -o ivf_mpi_omp_perf \
       -O2 -std=c++11 -fopenmp -I. -lm
if [ $? -ne 0 ]; then
    echo "[error] compilation failed, aborting."
    exit 1
fi
echo "[compile] done → ./ivf_mpi_omp_perf"

# Copy binary to ~/ann/ so qsub script can find it
cp ivf_mpi_omp_perf ~/ann/ivf_mpi_omp_perf

# ── Submit and wait ───────────────────────────────────────────────────────────
> test_perf.o
> test_perf.e

JOB_ID=$(qsub exp/qsub_mpi_omp_perf.sh)
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
echo "======== stderr (test_perf.e) ========"
cat test_perf.e
echo ""
echo "======== stdout (test_perf.o) ========"
cat test_perf.o
