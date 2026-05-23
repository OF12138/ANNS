#!/bin/bash
# =============================================================================
# exp/ivf_pthread_perf.sh — per-thread perf profiling for IVF_SIMD_CLUSTER_PTHREAD
#
# Compiles and runs ivf_pthread_perf.cc for two work-distribution strategies:
#   IVF_FLATTEN=0  cluster-split   (round-robin, non-uniform load)
#   IVF_FLATTEN=1  flatten-split   (even vector count per thread)
#
# Each variant is run at two nprobe values to show how load and cache behaviour
# scale with scan volume.
#
# Run from project root:
#   bash exp/ivf_pthread_perf.sh
#
# Optional: override nprobe values
#   NPROBES="8 16 64" bash exp/ivf_pthread_perf.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CXX="${CXX:-g++}"
CXXFLAGS="-O2 -std=c++11 -fopenmp -lpthread -I."
SRC="exp/ivf_pthread_perf.cc"

NPROBES="${NPROBES:-16 64}"

SEP="════════════════════════════════════════════════════════════"

echo "$SEP"
echo " IVF_SIMD_CLUSTER_PTHREAD — per-thread perf profiling"
echo " dataset: DEEP100K   threads=7   k=10"
echo " nprobe values: $NPROBES"
echo "$SEP"
echo ""

# Check perf permissions
paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)
if [ -n "$paranoid" ] && [ "$paranoid" -gt 2 ]; then
    echo "[warn] /proc/sys/kernel/perf_event_paranoid=$paranoid  (need <= 2)"
    echo "[warn] Hardware counters will be blocked; only vector counts shown."
    echo "[warn] To fix:  echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid"
    echo ""
fi

for flatten in 0 1; do
    bin="exp/ivf_pthread_perf_f${flatten}"

    echo "── Compiling IVF_FLATTEN=${flatten} ─────────────────────────────────"
    if ! $CXX $SRC -o "$bin" $CXXFLAGS -DIVF_FLATTEN=${flatten} 2>&1; then
        echo "  [ERROR] compilation failed, skipping."
        echo ""
        continue
    fi
    echo "  OK → $bin"
    echo ""

    for np in $NPROBES; do
        echo "── IVF_FLATTEN=${flatten}  nprobe=${np} ─────────────────────────────"
        "$bin" "$np"
        echo ""
    done
done

echo "$SEP"
echo " Done."
echo " Binaries: exp/ivf_pthread_perf_f0  exp/ivf_pthread_perf_f1"
echo " To rerun one:  ./exp/ivf_pthread_perf_f0 32"
echo "$SEP"
