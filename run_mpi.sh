#!/bin/bash
# Wrapper: compile main_mpi.cc, submit qsub_mpi.sh, wait, print output.
# Usage:  bash run_mpi.sh

# ── Compile ───────────────────────────────────────────────────────────────────
echo "[compile] mpicxx main_mpi.cc -o main_mpi -O2 -std=c++11 -lm"
mpicxx main_mpi.cc -o main_mpi -O2 -std=c++11 -lm
if [ $? -ne 0 ]; then
    echo "[error] compilation failed, aborting."
    exit 1
fi
echo "[compile] done."

# ── Submit and wait ───────────────────────────────────────────────────────────
> test.o
> test.e

JOB_ID=$(qsub qsub_mpi.sh)
if [ $? -ne 0 ]; then
    echo "[error] qsub failed: $JOB_ID"
    exit 1
fi
echo "[submit] job $JOB_ID"

# Poll qstat every 2 seconds until job status is C (completed) or disappears
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
echo "======== stderr (test.e) ========"
cat test.e
echo ""
echo "======== stdout (test.o) ========"
cat test.o
