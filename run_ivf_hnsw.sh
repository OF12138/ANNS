#!/bin/bash
# Compile main_ivf_hnsw.cc, submit PBS job, wait, print output.
# Usage: bash run_ivf_hnsw.sh

perl -i -pe 's/[^\x00-\x7F]//g' qsub_ivf_hnsw.sh

echo "[compile] g++ main_ivf_hnsw.cc -o main_ivf_hnsw ..."
g++ main_ivf_hnsw.cc -o main_ivf_hnsw -O2 -std=c++11 -lpthread -lm
if [ $? -ne 0 ]; then echo "[error] compilation failed."; exit 1; fi
echo "[compile] done."

cp main_ivf_hnsw ~/ann/main_ivf_hnsw

> test_ivf_hnsw.o
> test_ivf_hnsw.e

JOB_ID=$(qsub qsub_ivf_hnsw.sh)
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
echo "======== stderr (test_ivf_hnsw.e) ========"
cat test_ivf_hnsw.e
echo ""
echo "======== stdout (test_ivf_hnsw.o) ========"
cat test_ivf_hnsw.o
