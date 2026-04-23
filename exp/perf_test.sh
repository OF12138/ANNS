#!/bin/bash
# =============================================================================
# exp/perf_test.sh — build and run hardware-counter profiling for ANN algs
#
# Run from the project root:
#   bash exp/perf_test.sh
#
# Each algorithm is compiled into a separate binary under exp/ and run once.
# Results are printed to stdout; build/load noise goes to stderr.
#
# Requires: g++, linux/perf_event.h, /proc/sys/kernel/perf_event_paranoid <= 2
# =============================================================================

# Locate project root (works regardless of where the script is invoked from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CXX="${CXX:-g++}"
CXXFLAGS="-O2 -std=c++11 -fopenmp -lpthread -I."
SRC="exp/perf_exp.cc"

# Algorithm id : name pairs
ALG_IDS="3 4 6 7 9 11 12"
alg_name() {
    case $1 in
        3)  echo "FLAT_SIMD"        ;;
        4)  echo "FLAT_SIMD_UNROLL" ;;
        6)  echo "SQ_SIMD"          ;;
        7)  echo "SQ_SDC"           ;;
        9)  echo "PQ_RERANK"        ;;
        11) echo "PQ_CC_SIMD"       ;;
        12) echo "PQ_CC_UNROLL"     ;;
        *)  echo "ALG_$1"           ;;
    esac
}

SEP="════════════════════════════════════════════════════════════"

echo "$SEP"
echo " perf hardware counter experiments"
echo " dataset: DEEP100K  queries=2000  k=10  p=200"
echo "$SEP"
echo ""

for id in $ALG_IDS; do
    name=$(alg_name $id)
    bin="exp/perf_exp_${id}"

    # --- compile ---
    echo "── Compiling SEARCH_ALG=$id  ($name) ──────────────────────"
    if ! $CXX $SRC -o "$bin" $CXXFLAGS -DSEARCH_ALG=$id 2>&1; then
        echo "  [ERROR] compilation failed, skipping."
        echo ""
        continue
    fi

    # --- run ---
    echo "── Running $name ───────────────────────────────────────────"
    "$bin" 2>/dev/null
    echo ""
done

echo "$SEP"
echo " Done. Binaries: exp/perf_exp_{3,4,6,7,9,11,12}"
echo " To rerun one:   ./exp/perf_exp_6"
echo " To check perf permissions: cat /proc/sys/kernel/perf_event_paranoid"
echo "$SEP"
