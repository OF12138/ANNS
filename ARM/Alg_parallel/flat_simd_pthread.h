// =============================================================================
// flat_simd_pthread.h — Pthread-parallelized flat k-NN search (AArch64 NEON)
//
// Two independent parallelism strategies, both layered on top of the
// simd_inner_product_neon_unroll kernel from flat_simd.h:
//
// ── Strategy 1: Query-level parallelism ──────────────────────────────────────
// simd_flat_search_query_parallel(base, queries, ..., results, num_threads)
//
//   Divides the QUERY set into num_threads contiguous chunks.
//   Each thread independently calls simd_flat_search_unroll on its assigned
//   queries and writes results to non-overlapping slots in the output array.
//   No synchronization required during the search phase.
//
//   Thread model: static pool — threads created once for the whole batch.
//   Parallelism axis: across queries (embarrassingly parallel).
//   Effect on metrics: increases THROUGHPUT (queries/sec); individual per-query
//   wall-clock latency is unchanged (each query still runs on one thread).
//
// ── Strategy 2: Base-partition parallelism ────────────────────────────────────
// simd_flat_search_base_parallel(base, query, ..., num_threads)
//
//   For a SINGLE query, divides base[0..base_number) into num_threads equal
//   segments. Each thread scans its segment with NEON dot-product and maintains
//   a local max-heap of size k. After all threads join, the main thread merges
//   the local heaps into a global top-k result.
//
//   Thread model: dynamic — threads created and joined once per query call.
//   Parallelism axis: across base vectors (compute-bound scan, embarrassingly
//   parallel within one query).
//   Effect on metrics: reduces per-query LATENCY by ~num_threads× (minus
//   thread-create/join overhead, which is visible at small base_number).
//
// Correctness note (base-partition):
//   Each thread keeps its local top-k exactly. Since the true top-k globally
//   must appear in at least one segment's top-k, merging all local heaps and
//   taking the global top-k yields recall@k = 1.0 (exact, not approximate).
//
// Platform: AArch64. Compile: g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <pthread.h>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "flat_simd.h"   // simd_flat_search_unroll, simd_inner_product_neon_unroll


// =============================================================================
// Strategy 1 — Query-level parallelism
// =============================================================================

// Thread argument: each thread owns queries [q_start, q_end) exclusively.
struct _FlatQueryPArgs 
{
    float*  base;
    float*  queries;
    size_t  base_number;
    size_t  vecdim;
    size_t  k;
    int     q_start;   // inclusive
    int     q_end;     // exclusive
    std::priority_queue<std::pair<float, uint32_t>>* results; // shared array; thread owns [q_start,q_end)
};

// Thread function: calls simd_flat_search_unroll for every assigned query.
// No locks needed — each query index maps to a unique result slot.
static void* _flat_query_worker(void* arg)
{
    auto* a = static_cast<_FlatQueryPArgs*>(arg);
    for (int i = a->q_start; i < a->q_end; ++i)
        a->results[i] = simd_flat_search_unroll(
            a->base, a->queries + (size_t)i * a->vecdim,
            a->base_number, a->vecdim, a->k);
    return nullptr;
}

// simd_flat_search_query_parallel — batch k-NN, queries distributed across threads
//
// Parameters:
//   base        : base dataset [base_number × vecdim]
//   queries     : query batch  [num_queries × vecdim]
//   base_number : number of base vectors
//   vecdim      : vector dimensionality
//   k           : top-k results per query
//   num_queries : how many queries to process
//   results     : caller-allocated array of num_queries priority_queues (output)
//   num_threads : number of worker threads to spawn
//
// Algorithm:
//   Divides [0, num_queries) into num_threads chunks of size ceil(num_queries/num_threads).
//   Each thread handles one chunk; results are written directly to results[q_start..q_end).
//   Main thread calls pthread_create / pthread_join; no synchronization during compute.
//
// Timing note:
//   Measure wall time around this call to get total batch time.
//   avg_latency = wall_time / num_queries  (throughput metric, not latency reduction).
void simd_flat_search_query_parallel(
    float* base, float* queries,
    size_t base_number, size_t vecdim, size_t k,
    int num_queries,
    std::priority_queue<std::pair<float, uint32_t>>* results,
    int num_threads)
{
    std::vector<pthread_t>          tids(num_threads);
    std::vector<_FlatQueryPArgs>    args(num_threads);

    // Ceiling division: last thread may get fewer than chunk queries
    int chunk = (num_queries + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) 
    {
        args[t].base        = base;
        args[t].queries     = queries;
        args[t].base_number = base_number;
        args[t].vecdim      = vecdim;
        args[t].k           = k;
        args[t].q_start     = t * chunk;
        args[t].q_end       = std::min(num_queries, (t + 1) * chunk);
        args[t].results     = results;
        pthread_create(&tids[t], nullptr, _flat_query_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t)
        pthread_join(tids[t], nullptr);
}


// =============================================================================
// Strategy 2 — Base-partition parallelism
// =============================================================================

// Thread argument: each thread owns base vectors [base_start, base_end) exclusively.
// local_heap accumulates the thread's local top-k; read by main thread after join.
struct _FlatBasePArgs {
    float*  base;
    float*  query;
    size_t  base_start;  // inclusive
    size_t  base_end;    // exclusive
    size_t  vecdim;
    size_t  k;
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

// Thread function: linear scan over base[base_start..base_end) for one query.
// Uses simd_inner_product_neon_unroll (4-accumulator NEON) for dot products.
// Maintains a max-heap of size k; replaces worst when a closer vector is found.
static void* _flat_base_worker(void* arg)
{
    auto* a = static_cast<_FlatBasePArgs*>(arg);
    for (size_t i = a->base_start; i < a->base_end; ++i) 
    {
        float dis = 1.0f - simd_inner_product_neon_unroll(
            a->base + i * a->vecdim, a->query, a->vecdim);
        if (a->local_heap.size() < a->k) 
        {
            a->local_heap.push({dis, static_cast<uint32_t>(i)});
        } 
        else if (dis < a->local_heap.top().first) 
        {
            a->local_heap.push({dis, static_cast<uint32_t>(i)});
            a->local_heap.pop();
        }
    }
    return nullptr;
}

// simd_flat_search_base_parallel — single-query k-NN, base split across threads
//
// Parameters:
//   base        : base dataset [base_number × vecdim]
//   query       : single query vector [vecdim]
//   base_number : number of base vectors
//   vecdim      : vector dimensionality
//   k           : top-k results to return
//   num_threads : number of worker threads to spawn
//
// Algorithm:
//   1. Divide base[0..base_number) into num_threads equal segments.
//   2. Create num_threads threads; thread t scans base[t*chunk..(t+1)*chunk)
//      and builds a local max-heap of at most k candidates.
//   3. Join all threads (barrier).
//   4. Merge: iterate all local heaps (≤ num_threads × k candidates total),
//      maintain a global max-heap of size k — the final top-k.
//
// Correctness:
//   The true nearest neighbor of any query must appear in exactly one segment.
//   That segment's thread keeps it in its local top-k.  Merging all local top-k
//   sets therefore yields the global top-k exactly (recall@k = 1.0).
//
// Overhead: pthread_create + pthread_join for num_threads threads per call.
// On DEEP100K (100K vectors, 96 dims) this overhead is measurable (~5–30 µs);
// speedup requires the scan time to dominate (it does at full base_number).
//
// Returns: max-heap of (IP distance, vector index) pairs — same as flat_search.
std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_base_parallel(
    float* base, float* query,
    size_t base_number, size_t vecdim, size_t k,
    int num_threads)
{
    std::vector<pthread_t>        tids(num_threads);
    std::vector<_FlatBasePArgs>   args(num_threads);

    // Ceiling division: last thread may get a smaller segment
    size_t chunk = (base_number + (size_t)num_threads - 1) / (size_t)num_threads;
    for (int t = 0; t < num_threads; ++t) {
        args[t].base       = base;
        args[t].query      = query;
        args[t].base_start = (size_t)t * chunk;
        args[t].base_end   = std::min(base_number, (size_t)(t + 1) * chunk);
        args[t].vecdim     = vecdim;
        args[t].k          = k;
        // local_heap default-constructed (empty)
        pthread_create(&tids[t], nullptr, _flat_base_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t)
        pthread_join(tids[t], nullptr);

    // ── Merge phase ───────────────────────────────────────────────────────────
    // At most num_threads * k candidates; take the global top-k.
    // A max-heap of size k: evict the farthest whenever we exceed k.
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (int t = 0; t < num_threads; ++t) {
        while (!args[t].local_heap.empty()) {
            auto top = args[t].local_heap.top();
            args[t].local_heap.pop();
            if (result.size() < k) {
                result.push(top);
            } else if (top.first < result.top().first) {
                result.push(top);
                result.pop();
            }
        }
    }
    return result;
}
