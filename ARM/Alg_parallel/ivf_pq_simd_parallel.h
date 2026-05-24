// =============================================================================
// ivf_pq_simd_parallel.h — Query-parallel batch search for IVF-PQ (IVF-first)
//
// Implements query-level parallelism for ivfpq_search_simd (SEARCH_ALG 26):
//
//   ivfpq_batch_search_pthread (SEARCH_ALG 28)
//     Static query partition across Pthreads.
//     Divides [0, num_queries) into num_threads contiguous equal-size chunks.
//     Each thread independently calls ivfpq_search_simd for its slice:
//       Thread 0: queries[0   .. Q/T)
//       Thread 1: queries[Q/T .. 2Q/T)
//       ...
//     No shared writes; no locks; no synchronization during search.
//     Metric: batch throughput (queries/sec). Single-query latency unchanged.
//
// Why no pre-flatten:
//   Flattening the cluster vectors before splitting would only benefit
//   intra-query fine-scan parallelism (each query scans ~3K vectors across
//   nprobe clusters). Inter-query parallelism needs no flattening — each
//   thread runs the complete per-query pipeline on its own slice of queries.
//
// Platform: AArch64.
// Compile:  g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <pthread.h>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include "ivf_pq_simd.h"


// =============================================================================
// _IVFPQBatchArgs — per-thread argument block
// =============================================================================
struct _IVFPQBatchArgs {
    const IVFPQIndex* idx;
    const float*      base;
    const float*      queries;
    int               start;    // first query index (inclusive)
    int               end;      // last query index (exclusive)
    size_t            vecdim;
    size_t            k;
    size_t            nprobe;
    size_t            p;
    std::priority_queue<std::pair<float, uint32_t>>* results;
};

// =============================================================================
// _ivfpq_batch_worker — thread entry: process queries [start, end)
// =============================================================================
static void* _ivfpq_batch_worker(void* arg)
{
    auto* a = static_cast<_IVFPQBatchArgs*>(arg);
    for (int i = a->start; i < a->end; ++i) {
        a->results[i] = ivfpq_search_simd(
            *a->idx, a->base,
            a->queries + (size_t)i * a->vecdim,
            a->k, a->nprobe, a->p);
    }
    return nullptr;
}


// =============================================================================
// ivfpq_batch_search_pthread — static query-parallel IVF-PQ batch search
//
// Parameters:
//   idx         — built IVFPQIndex, read-only, shared across threads
//   base        — original float base vectors for exact rerank, read-only
//   queries     — flat query array [num_queries × vecdim]
//   num_queries — total number of queries
//   vecdim      — vector dimension
//   k           — final top-k results per query
//   nprobe      — IVF clusters to scan per query (latency-recall knob)
//   p           — rerank candidate pool size (p ≥ k)
//   results     — output array [num_queries], pre-allocated by caller
//   num_threads — number of Pthread worker threads
// =============================================================================
void ivfpq_batch_search_pthread(
    const IVFPQIndex& idx, const float* base,
    const float* queries, int num_queries, size_t vecdim,
    size_t k, size_t nprobe, size_t p,
    std::priority_queue<std::pair<float, uint32_t>>* results,
    int num_threads)
{
    const int T = std::min(num_threads, num_queries);

    std::vector<pthread_t>       tids(T);
    std::vector<_IVFPQBatchArgs> args(T);

    // Static partition: distribute queries as evenly as possible
    int base_chunk = num_queries / T;
    int remainder  = num_queries % T;

    int start = 0;
    for (int t = 0; t < T; ++t) {
        int chunk  = base_chunk + (t < remainder ? 1 : 0);
        args[t]    = { &idx, base, queries, start, start + chunk,
                       vecdim, k, nprobe, p, results };
        start     += chunk;
        pthread_create(&tids[t], nullptr, _ivfpq_batch_worker, &args[t]);
    }

    for (int t = 0; t < T; ++t)
        pthread_join(tids[t], nullptr);
}
