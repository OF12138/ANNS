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

echo "[submit] qsub qsub_mpi.sh"
qsub -W block=true qsub_mpi.sh
echo ""
echo "======== stderr (test.e) ========"
cat test.e
echo ""
echo "======== stdout (test.o) ========"
cat test.o
